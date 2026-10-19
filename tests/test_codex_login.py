"""Authentication and process-boundary regressions for optional Codex CLI sessions."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config, install
from harness2.core.proposer import Proposer, ProposerError
from harness2.core.spec import ChannelSpec, resolve_channel_model
from harness2.substrate.codex import CodexSubstrate


class CodexLoginTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.substrate = CodexSubstrate()
        for name, value in (
            ("CODEX_AUTH_MODE", "chatgpt"),
            ("CODEX_BIN", "codex"),
            ("CODEX_LOGIN_HOME", ""),
        ):
            override = patch.object(config, name, value, create=True)
            override.start()
            self.addCleanup(override.stop)

    def test_api_mode_keeps_pinned_binary_generated_home_and_flags(self):
        with patch.object(config, "CODEX_AUTH_MODE", "api"):
            argv = self.substrate.exec_argv("task", self.root, "model")
            env = self.substrate.exec_env(
                {"CODEX_HOME": "/existing", "OPENAI_BASE_URL": "https://unused"},
                codex_home=self.root,
            )
        self.assertEqual(argv[0], str(Path(config.CODEX_DIST) / "codex"))
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertEqual(env["CODEX_HOME"], str(self.root))
        self.assertNotIn("OPENAI_BASE_URL", env)

    def test_chatgpt_uses_existing_login_home_without_api_credentials(self):
        original = {
            "CODEX_HOME": "/existing/login",
            "PATH": "/bin",
            "OPENAI_API_KEY": "sentinel",
            "CODEX_API_KEY": "sentinel",
            "OPENAI_BASE_URL": "https://unused",
        }
        env = self.substrate.exec_env(original, codex_home=self.root)
        self.assertEqual(env["CODEX_HOME"], "/existing/login")
        self.assertEqual(env["PATH"], "/bin")
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
            self.assertNotIn(key, env)
        self.assertIn("OPENAI_API_KEY", original)
        argv = self.substrate.exec_argv("task", self.root, "model")
        self.assertEqual(argv[0], shutil.which("codex") or "codex")
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("workspace-write", argv)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertIn("--ephemeral", argv)

    def test_chatgpt_home_override_does_not_create_a_directory(self):
        login_home = self.root / "saved-login"
        with patch.object(config, "CODEX_LOGIN_HOME", str(login_home)):
            env = self.substrate.exec_env({}, codex_home=self.root / "api")
        self.assertEqual(env["CODEX_HOME"], str(login_home))
        self.assertFalse(login_home.exists())

    def test_chatgpt_setup_and_preflight_require_saved_chatgpt_login(self):
        with (
            patch("shutil.which", return_value="/bin/codex"),
            patch("subprocess.run", return_value=Mock(
                returncode=0, stdout="Logged in using ChatGPT\n", stderr=""
            )) as run,
            patch.object(config, "CODEX_BASE_URL", ""),
            patch.object(config, "VERTEX_PROXY_KEY", ""),
        ):
            self.substrate.preflight()
            install.configure_codex()
            self.assertIsNone(install.codex_config_error())
        self.assertEqual(run.call_args.args[0], ["/bin/codex", "login", "status"])
        for result in (
            Mock(returncode=1, stdout="", stderr="Not logged in"),
            Mock(returncode=0, stdout="Logged in using an API key", stderr=""),
        ):
            with (
                patch("shutil.which", return_value="/bin/codex"),
                patch("subprocess.run", return_value=result),
                self.assertRaisesRegex(RuntimeError, "codex login"),
            ):
                self.substrate.preflight()

    def test_invalid_mode_and_missing_cli_fail_before_a_model_call(self):
        with (
            patch.object(config, "CODEX_AUTH_MODE", "typo"),
            self.assertRaisesRegex(config.ConfigError, "HARNESS2_CODEX_AUTH_MODE"),
        ):
            self.substrate.preflight()
        with (
            patch("shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "Codex CLI"),
        ):
            self.substrate.preflight()

    def test_api_preflight_still_requires_endpoint_and_key(self):
        with (
            patch.object(config, "CODEX_AUTH_MODE", "api"),
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(config.ConfigError, "HARNESS2_CODEX_BASE_URL"),
        ):
            self.substrate.preflight()


class CodexProposerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_proposer(self):
        return Proposer(ChannelSpec("codex_cli", "test-model"), audit_dir=self.root / "audit")

    def test_codex_channel_resolves_a_bare_model_without_changing_defaults(self):
        self.assertEqual(resolve_channel_model("codex_cli", "test-model").model, "test-model")
        self.assertEqual(config.DEFAULT_PROPOSER_CHANNEL, "opencode")
        with self.assertRaises(ValueError):
            resolve_channel_model("codex_cli", "provider/model")

    def test_success_harvests_edits_and_cached_calls_do_not_run_again(self):
        source = self.root / "source"
        (source / "harness").mkdir(parents=True)
        (source / "harness/AGENTS.md").write_text("before\n")
        visited = []

        def execute(argv, **kwargs):
            cwd = Path(kwargs["cwd"])
            visited.append(cwd)
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            (cwd / "context/harness/AGENTS.md").write_text("after\n")
            answer = Path(argv[argv.index("--output-last-message") + 1])
            answer.write_text("Improved the instructions.\n")
            return Mock(returncode=0, stdout='{"type":"turn.completed"}\n', stderr="")

        proposer = self.make_proposer()
        with patch("subprocess.run", side_effect=execute) as run:
            reply = proposer.call("Improve this", source, "edit", harvest="context/harness")
            cached = proposer.call("Improve this", source, "edit", harvest="context/harness")
        self.assertEqual(run.call_count, 1)
        self.assertTrue(cached.cached)
        self.assertEqual((reply.workspace / "AGENTS.md").read_text(), "after\n")
        self.assertEqual((source / "harness/AGENTS.md").read_text(), "before\n")
        self.assertFalse(visited[0].exists())
        metadata = json.loads(reply.audit.with_name(reply.audit.name.replace(
            ".response.md", ".meta.json"
        )).read_text())
        self.assertEqual(metadata["channel"], "codex_cli")

    def test_nonzero_exit_rejects_partial_output_and_does_not_cache_success(self):
        proposer = self.make_proposer()

        def execute(argv, **kwargs):
            Path(argv[argv.index("--output-last-message") + 1]).write_text("partial")
            return Mock(returncode=1, stdout="", stderr="rate limit")

        with (
            patch("subprocess.run", side_effect=execute),
            self.assertRaisesRegex(ProposerError, "rate limit"),
        ):
            proposer.call("work", None, "failed", max_attempts=1)
        self.assertFalse(proposer._calls[-1].ok)

    def test_timeout_and_empty_output_do_not_become_successful_replies(self):
        proposer = self.make_proposer()
        for result in (
            subprocess.TimeoutExpired("codex", 1),
            Mock(returncode=0, stdout='{"type":"turn.completed"}', stderr=""),
        ):
            kwargs = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
            with patch("subprocess.run", **kwargs), self.assertRaises(ProposerError):
                proposer.call("work", None, "failed", max_attempts=1)

    def test_terminal_failure_rejects_a_written_final_answer(self):
        def execute(argv, **kwargs):
            Path(argv[argv.index("--output-last-message") + 1]).write_text("partial")
            return Mock(returncode=0, stdout=json.dumps({
                "type": "turn.failed", "error": {"message": "quota exhausted"}
            }), stderr="")

        with patch("subprocess.run", side_effect=execute), self.assertRaisesRegex(
            ProposerError, "quota exhausted"
        ):
            self.make_proposer().call("work", None, "failed", max_attempts=1)

    def test_cache_is_not_shared_between_models(self):
        def execute(argv, **kwargs):
            Path(argv[argv.index("--output-last-message") + 1]).write_text("done")
            return Mock(returncode=0, stdout='{"type":"turn.completed"}', stderr="")

        with patch("subprocess.run", side_effect=execute) as run:
            for model in ("model-one", "model-two"):
                proposer = Proposer(ChannelSpec("codex_cli", model), audit_dir=self.root / "audit")
                proposer.call("same task", None, "same-label")
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
