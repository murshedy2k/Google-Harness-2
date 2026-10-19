# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for JobBench OpenCode render-directory cleanup."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2.benchmarks import jobbench
from harness2.substrate.base import Install, RunRequest


class RenderCleanupTests(unittest.TestCase):
    def _stub(self, root: Path, *, fail: bool = False):
        class Substrate:
            name = "oc"

            def preflight(self):
                pass

        class Stub:
            substrate = Substrate()
            store = root
            rendered = None

            def _launch_common(self, req):
                return req.task_ref, req.run_ids[0].replace("/", "__"), int(req.timeout)

            def domain_of(self, task_id):
                return task_id.split("/", 1)[0]

            def _run_runner(self, runner, env, *, wall, label):
                self.rendered = Path(env["HARNESS_RENDER_DIR"])
                if not self.rendered.is_dir():
                    raise AssertionError("render directory was not created for the runner")
                if fail:
                    raise RuntimeError("runner failed")

            def _require_artifacts(self, task_id, label):
                return root / "out"

        return Stub()

    def _call(self, stub, root):
        install = Install(staging_dir=root / "staging", fingerprint="x")
        req = RunRequest(
            task_ref="prof/task-1",
            run_ids=("ns/prof/task-1/base1",),
            timeout=5,
        )
        with patch.object(jobbench, "render_opencode_tree", lambda *a, **k: None):
            return jobbench.JobBenchAdapter._launch_opencode(stub, req, install)

    def test_render_dir_removed_after_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stub = self._stub(root)
            self.assertEqual(self._call(stub, root), [root / "out"])
            self.assertIsNotNone(stub.rendered)
            self.assertFalse(stub.rendered.exists())

    def test_render_dir_removed_when_runner_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stub = self._stub(root, fail=True)
            with self.assertRaises(RuntimeError):
                self._call(stub, root)
            self.assertIsNotNone(stub.rendered)
            self.assertFalse(stub.rendered.exists())


if __name__ == "__main__":
    unittest.main()
