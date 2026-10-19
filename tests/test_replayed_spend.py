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

"""Improver spend replayed from the audit cache is reported (issue #50)."""

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from harness2.cli import _common as C
from harness2.cli import report
from harness2.cli.report import _spend_and_fallbacks
from harness2.core import judge, recursion, schedule, tree
from harness2.core.proposer import Proposer
from harness2.demo import BASE_SCRIPT, DemoAdapter

COST = 0.25
NOTE = (
    "  of which $1.00 was replayed from the audit cache: paid when those sessions first ran, "
    "possibly in an earlier sweep\n"
)
REFLECTION = (
    "<edits>\ncomponent: scripts/summarize.py\ncandidates: inv1\nchange: c\nobserved: o\n"
    "verdict: helped\n</edits>\n<experience>\ncue: c\nlesson: l scripts/summarize.py\n"
    "evidence: e\nconfidence: low\n</experience>"
)


def broken(harness):
    (harness / "scripts/summarize.py").write_text("def broken(:\n")


def vanish(harness):
    shutil.rmtree(harness)


def valid(harness):
    (harness / "scripts/summarize.py").write_text(BASE_SCRIPT.replace(" if amount > 0", ""))


@dataclass
class ScriptedProposer(Proposer):
    """Only the model transport is scripted; cache and audit are real."""

    edits: list = field(default_factory=list)
    choose: bool = True

    def _opencode(self, cwd, mission, system, wall):
        context = cwd / "context"
        if (context / "base_harness").is_dir():
            return REFLECTION, COST, ""
        if (context / "candidates").is_dir():
            chosen = sorted((context / "candidates").glob("*.md"))[0].stem
            choice = f"\n```choice\n{chosen}\n```" if self.choose else ""
            return "<why>scripted</why>" + choice, COST, ""
        if (context / "harness").is_dir():
            (self.edits.pop(0) if self.edits else valid)(context / "harness")
        return "<notes>scripted edit</notes>", COST, ""


class ReplayedSpendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = DemoAdapter(self.root)

    def proposer(self, name="audit", **kwargs):
        channel = self.adapter.spec.channels["proposer"]
        return ScriptedProposer(channel, audit_dir=self.root / name, **kwargs)

    def run_stream(self, prop, k=1, modes=("agg",)):
        schedule.run_domain(
            "accounting",
            ["ledger-1"],
            k,
            adapter=self.adapter,
            prop=prop,
            runs_root=C.runs_root_for(self.adapter),
            library_root=self.root / "library",
            modes=modes,
        )
        return _spend_and_fallbacks(self.summary(k).parents[1], modes, k)[1]

    def summary(self, k=1):
        namespace = C.summary_namespace(self.adapter, k, seed=None)
        return C.runs_root_for(self.adapter) / namespace / "ledger-1/summary.json"

    def interrupt(self, k=1):
        path = self.summary(k)
        path.write_text(json.dumps({**json.loads(path.read_text()), "status": "partial"}))

    def render_report(self):
        with patch.object(C, "make_adapter", return_value=self.adapter):
            with redirect_stdout(io.StringIO()) as out:
                report.main(["--bench", "jb", "--k", "1"])
        return out.getvalue()

    def run_stage(self, prop):
        return recursion._run_stage(
            label="inv1",
            dim="INV",
            job="Repair the script",
            adapter=self.adapter,
            prop=prop,
            ctx_dir=self.root / "context",
            task_id="ledger-1",
            rollouts=[],
            fallback=tree.read_tree(self.adapter.substrate.surface(), self.adapter.base_harness()),
            harness_dir=self.adapter.base_harness(),
            library_dir=None,
        )

    def test_cached_reply_reports_original_cost_without_claiming_new_spend(self):
        prop = self.proposer()
        live = prop.call("mission", None, "inv1")
        replayed = prop.call("mission", None, "inv1")
        self.assertEqual((live.cached, live.cost_usd, live.replayed_usd), (False, COST, 0.0))
        self.assertEqual(
            (replayed.cached, replayed.cost_usd, replayed.replayed_usd), (True, 0.0, COST)
        )
        self.assertEqual(replayed.text, live.text)

    def test_untrusted_cost_record_still_replays_the_reply_at_zero(self):
        cases = {
            "missing": None,
            "bad json": "{not json",
            "not a dict": "[1]",
            "no cost": {"ok": True},
            "not ok": {"ok": False, "cost_usd": COST},
            "nan": {"ok": True, "cost_usd": float("nan")},
            "infinity": {"ok": True, "cost_usd": float("inf")},
            "negative": {"ok": True, "cost_usd": -COST},
            "bool": {"ok": True, "cost_usd": True},
            "huge int": {"ok": True, "cost_usd": 10**400},
        }
        for name, meta in cases.items():
            with self.subTest(name):
                prop = self.proposer(name)
                live = prop.call("mission", None, "inv1")
                (path,) = prop.audit_dir.glob("*.meta.json")
                if meta is None:
                    path.unlink()
                else:
                    path.write_text(meta if isinstance(meta, str) else json.dumps(meta))
                replayed = prop.call("mission", None, "inv1")
                self.assertEqual(
                    (replayed.cached, replayed.text, replayed.replayed_usd), (True, live.text, 0.0)
                )

    def test_report_spend_includes_replayed_cost_and_leaves_old_summaries_unchanged(self):
        def write(ns, stages, selection):
            task = self.root / ns / "ledger-1"
            task.mkdir(parents=True)
            record = {"modes": {"agg": {"stages": stages}}, "selections": {"agg": selection}}
            (task / "summary.json").write_text(json.dumps(record))

        write(
            "rerun",
            [{"usd": 0.0, "replayed_usd": COST}, {"usd": COST, "replayed_usd": COST}],
            {"usd": 0.0, "replayed_usd": COST},
        )
        write("old", [{"usd": COST}, {"usd": COST}], {"usd": COST})
        rerun = _spend_and_fallbacks(self.root / "rerun", ("agg",), 1)[1]
        old = _spend_and_fallbacks(self.root / "old", ("agg",), 1)[1]
        self.assertEqual((rerun, old), ({("agg", 1): 4 * COST}, {("agg", 1): 3 * COST}))

    def test_stage_reports_replayed_and_live_cost_separately(self):
        prop = self.proposer(edits=[broken, vanish, valid])
        first = self.run_stage(prop)
        second = self.run_stage(prop)
        self.assertEqual((first.fell_back, first.usd, first.replayed_usd), (True, 2 * COST, 0.0))
        self.assertEqual((second.fell_back, second.usd, second.replayed_usd), (False, COST, COST))

    def test_fallback_stage_keeps_the_replayed_cost(self):
        prop = self.proposer(edits=[broken, vanish, vanish])
        self.run_stage(prop)
        second = self.run_stage(prop)
        self.assertEqual((second.fell_back, second.usd, second.replayed_usd), (True, COST, COST))

    def test_rerun_after_interruption_reports_the_same_improver_spend(self):
        prop = self.proposer()
        first = self.run_stream(prop)
        self.interrupt()
        rerun = self.run_stream(prop)
        self.interrupt()
        again = self.run_stream(prop)
        resumed = self.run_stream(prop)
        self.assertEqual((first, rerun, again, resumed), ({("agg", 1): 4 * COST},) * 4)
        stages = json.loads(self.summary().read_text())["modes"]["agg"]["stages"]
        self.assertEqual([s["usd"] for s in stages], [0.0, 0.0, 0.0])

    def test_rerun_with_k2_counts_the_replayed_selector_for_each_mode(self):
        prop = self.proposer()
        first = self.run_stream(prop, k=2, modes=("agg", "seq"))
        self.interrupt(k=2)
        rerun = self.run_stream(prop, k=2, modes=("agg", "seq"))
        self.assertEqual(first, {("agg", 2): 8 * COST, ("seq", 2): 8 * COST})
        self.assertEqual(rerun, first)

    def test_replayed_selector_reply_without_a_choice_still_counts(self):
        prop = self.proposer(choose=False)
        first = self.run_stream(prop, k=2, modes=("agg", "seq"))
        rerun = self.run_stream(prop, k=2, modes=("agg", "seq"))
        self.assertEqual(first, {("agg", 2): 7 * COST, ("seq", 2): 7 * COST})
        self.assertEqual(rerun, first)

    def test_ledger_records_replayed_cost_and_keeps_live_spend_and_budget(self):
        prop = self.proposer()
        self.run_stream(prop)
        self.interrupt()
        self.run_stream(prop)
        ledger = json.loads(self.summary().read_text())["modes"]["agg"]["ledger"]
        self.assertEqual(
            (ledger["replayed_usd"], ledger["usd"], ledger["R"], ledger["P"]), (3 * COST, 0.0, 4, 3)
        )

    def test_report_footer_notes_replayed_spend_only_when_there_is_some(self):
        prop = self.proposer()
        self.run_stream(prop)
        chosen = json.loads(self.summary().read_text())["selections"]["agg"]
        row = judge.Scored(
            "ledger-1", "agg", 1, chosen["chosen_label"], chosen["chosen_run_id"], False, 1, 1
        )
        judge.write([row], self.summary().parents[1] / "judged.json")
        first = self.render_report()
        self.interrupt()
        self.run_stream(prop)
        rerun = self.render_report()
        self.assertTrue(first.endswith("spend across reported modes: $1.00\n"), first)
        self.assertNotIn("replayed", first)
        self.assertEqual(rerun, first + NOTE)

    def test_report_footer_skips_replayed_spend_below_one_cent(self):
        ns = self.summary().parents[1]
        (ns / "ledger-1").mkdir(parents=True)
        stages = [{"usd": 0.0, "replayed_usd": 0.004}]
        record = {"modes": {"agg": {"stages": stages}}, "selections": {"agg": {"usd": 0.0}}}
        (ns / "ledger-1/summary.json").write_text(json.dumps(record))
        judge.write(
            [judge.Scored("ledger-1", "agg", 1, "agg1", "run", False, 1, 1)], ns / "judged.json"
        )
        out = self.render_report()
        self.assertNotIn("spend across", out)
        self.assertNotIn("replayed", out, out)


if __name__ == "__main__":
    unittest.main()
