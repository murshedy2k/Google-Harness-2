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

"""Render a self-contained HTML dashboard from a local harness2 run directory.

The dashboard is generated from the on-disk conventions written by run_sweep and
harness2-demo (see harness2/cli/_common.py): per-task rollout executions, their
answered totals, the selection candidates per stage, and the adopted EXPERIENCE
library. It has zero dependencies, zero JavaScript, and no network access, so
the output file can be shared or committed as-is.

Usage:

    harness2-dashboard --run runs/demo            # writes runs/demo/dashboard.html
    harness2-dashboard --run runs/demo --open     # also opens it in a browser
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import webbrowser
from collections import Counter
from pathlib import Path

from ._common import HelpFormatter

_TOTAL_RE = re.compile(r"Total:\s*(-?[\d,]+(?:\.\d+)?)")
_CAND_TOTAL_RE = re.compile(r"-(\d+)\.md$")
_TRAJ_CHARS = 1200
_ANSWER_CHARS = 400
_LIB_CHARS = 1000

_STYLE = """
:root { color-scheme: light dark; }
body { font: 14px/1.55 system-ui, sans-serif; margin: 0; padding: 24px;
       background: #f6f7f9; color: #1a1d21; }
main { max-width: 980px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 8px; border-bottom: 1px solid #d8dce1;
     padding-bottom: 6px; text-transform: uppercase; letter-spacing: .04em; }
h3 { font-size: 14px; margin: 18px 0 8px; }
.sub { color: #5f6771; margin: 0 0 18px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.chip { border-radius: 999px; padding: 3px 11px; font-size: 12.5px; font-weight: 600;
        border: 1px solid #cfd4da; background: #fff; color: #2a2f36; white-space: nowrap; }
.chip.base { background: #eceff2; }
.chip.ok { border-color: #1a7f37; background: #e6f4ea; color: #116329; }
.chip.miss { border-color: #c93c37; background: #fdecea; color: #a5211c; }
.card { background: #fff; border: 1px solid #d8dce1; border-radius: 10px;
        padding: 14px 16px; margin: 10px 0; }
.card summary { cursor: pointer; font-weight: 600; }
pre { background: #f2f4f6; border: 1px solid #dde1e6; border-radius: 8px;
      padding: 10px 12px; overflow: auto; font-size: 12.5px; white-space: pre-wrap; }
.muted { color: #5f6771; }
.badge { font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 6px;
         vertical-align: 1px; }
.badge.ref { background: #1a7f37; color: #fff; }
@media (prefers-color-scheme: dark) {
  body { background: #16181b; color: #e4e7ea; }
  .card, .chip { background: #1e2126; border-color: #33383f; color: #d7dbdf; }
  .chip.base { background: #262a30; }
  pre { background: #101216; border-color: #2c3138; }
  h2 { border-color: #33383f; }
}
"""


def _read(path: Path, limit: int | None = None) -> str:
    """Read a text file defensively: missing/unreadable files degrade to ''."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text[:limit] if limit and len(text) > limit else text


def _stage_of(rollout: str) -> str:
    """Rollout dir names look like base1, agg1, inv1, c1, dom1; group by prefix."""
    match = re.match(r"[a-z]+", rollout)
    return match.group(0) if match else rollout


def _parse_total(text: str) -> str | None:
    """Pull the answered 'Total: N' value out of an answer or trajectory text."""
    match = _TOTAL_RE.search(text)
    return match.group(1).replace(",", "") if match else None


def _collect_rollouts(run_root: Path, mode: str, task: str) -> list[dict[str, str]]:
    """One record per executed rollout under executions/<mode>/<task>/<rollout>/."""
    base = run_root / "executions" / mode / task
    if not base.is_dir():
        return []
    rollouts = []
    for name in sorted(p.name for p in base.iterdir() if p.is_dir()):
        directory = base / name
        answer = _read(directory / "answer.txt", _ANSWER_CHARS)
        trajectory = _read(directory / "trajectory.md", _TRAJ_CHARS)
        total = _parse_total(answer) or _parse_total(trajectory)
        rollouts.append(
            {
                "name": name,
                "stage": _stage_of(name),
                "total": total,
                "answer": answer,
                "trajectory": trajectory,
            }
        )
    return rollouts


def _collect_candidates(run_root: Path, mode: str, task: str) -> list[dict[str, str]]:
    """Selection candidates per stage; candidate filenames embed the total."""
    out = []
    pattern = "*/candidates/*.md"
    for path in sorted((run_root / "runs" / mode / task / "selection").glob(pattern)):
        match = _CAND_TOTAL_RE.search(path.name)
        out.append(
            {
                "file": path.relative_to(run_root).as_posix(),
                "stage": path.relative_to(run_root).as_posix().split("/")[3],
                "total": match.group(1) if match else None,
            }
        )
    return out


def _reference(rollouts: list[dict[str, str]], selection: list[dict[str, str]]) -> str | None:
    """The value the ledgers are graded against: a selection total, else the
    majority among non-base rollouts."""
    for candidate in selection:
        if candidate["total"]:
            return candidate["total"]
    counts = Counter(
        r["total"] for r in rollouts if r["total"] and not r["name"].startswith("base")
    )
    if counts:
        return counts.most_common(1)[0][0]
    return None


def _collect_library(run_root: Path) -> list[dict[str, str]]:
    """Every adopted EXPERIENCE.md under the run's top-level library directory.

    Rollout-context libraries (runs/<task>/context/<stage>/library/...) hold
    candidate knowledge that was not selected; they are intentionally excluded.
    """
    library = run_root / "library"
    if not library.is_dir():
        return []
    return [
        {
            "file": path.relative_to(run_root).as_posix(),
            "text": _read(path, _LIB_CHARS),
        }
        for path in sorted(library.rglob("EXPERIENCE.md"))
    ]


def _summary(run_root: Path) -> dict:
    """Stats from the run's top-level summary JSON, if any (demo.json).

    Any JSON with a 'tasks' list qualifies; extra JSON files next to it (drafts,
    exported data) are skipped, so the scan stays tolerant of clutter.
    """
    for path in sorted(run_root.glob("*.json")):
        try:
            data = json.loads(_read(path) or "{}")
        except json.JSONDecodeError:
            continue
        tasks = data.get("tasks") if isinstance(data, dict) else None
        if not isinstance(tasks, list):
            continue
        done = sum(1 for t in tasks if t.get("status") == "ok")
        counts = []
        for field in ("execution_count", "editor_calls"):
            total = _int(data.get(field)) or sum(_int(t.get(field)) for t in tasks)
            counts.append(total)
        return {
            "file": path.name,
            "done": done,
            "failed": len(tasks) - done,
            "executions": counts[0],
            "editor_calls": counts[1],
        }
    return {}


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def collect(run_root: Path) -> dict:
    """Walk a run directory and return the plain-data model for the renderer."""
    runs = run_root / "runs"
    modes = []
    if runs.is_dir():
        for mode in sorted(p.name for p in runs.iterdir() if p.is_dir()):
            tasks = []
            for task in sorted(p.name for p in (runs / mode).iterdir() if p.is_dir()):
                rollouts = _collect_rollouts(run_root, mode, task)
                if not rollouts:
                    continue
                selection = _collect_candidates(run_root, mode, task)
                tasks.append(
                    {
                        "name": task,
                        "rollouts": rollouts,
                        "selection": selection,
                        "reference": _reference(rollouts, selection),
                    }
                )
            if tasks:
                modes.append({"name": mode, "tasks": tasks})
    return {
        "run_root": run_root.name,
        "modes": modes,
        "summary": _summary(run_root),
        "library": _collect_library(run_root),
    }


def _chip_class(rollout: dict[str, str], reference: str | None) -> str:
    if rollout["name"].startswith("base"):
        return "chip base"
    if reference and rollout["total"] == reference:
        return "chip ok"
    if rollout["total"] is None:
        return "chip miss"
    return "chip"


def _summary_chips(summary: dict, executions: int, entries: int) -> str:
    if summary:
        parts = [
            f"{summary['done']}/{summary['done'] + summary['failed']} tasks ok",
            f"{summary['executions']} executions",
            f"{summary['editor_calls']} editor calls",
        ]
    else:
        parts = [f"{executions} executions"]
    parts.append(f"{entries} library entries")
    return "".join(f'<span class="chip base">{html.escape(p)}</span>' for p in parts)


def render(data: dict) -> str:
    """Render the collected model into a complete, self-contained HTML page."""
    modes = data["modes"]
    executions = sum(len(t["rollouts"]) for m in modes for t in m["tasks"])
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>harness2 dashboard</title><style>" + _STYLE + "</style></head><body><main>",
        f"<h1>harness2 dashboard — {html.escape(data['run_root'])}</h1>",
        "<p class='sub'>Generated by <code>harness2-dashboard</code> · "
        "self-contained, no JavaScript.</p>",
        f"<div class='chips'>{_summary_chips(data['summary'], executions, len(data['library']))}</div>",
    ]
    for mode in modes:
        parts.append(f"<h2>mode: {html.escape(mode['name'])}</h2>")
        for task in mode["tasks"]:
            ref = task["reference"]
            ref_label = ref if ref is not None else "n/a"
            parts.append(
                f"<h3>{html.escape(task['name'])} "
                f"<span class='muted'>(reference: {html.escape(ref_label)})</span></h3>"
            )
            parts.append("<div class='chips'>")
            for rollout in task["rollouts"]:
                label = rollout["name"]
                if rollout["total"] is not None:
                    label += f": {rollout['total']}"
                badge = (
                    "<span class='badge ref'>✓ ref</span>"
                    if ref is not None and rollout["total"] == ref
                    and not rollout["name"].startswith("base")
                    else ""
                )
                parts.append(
                    f"<span class='{_chip_class(rollout, ref)}'>{html.escape(label)}{badge}</span>"
                )
            parts.append("</div>")
            for rollout in task["rollouts"]:
                if not rollout["trajectory"]:
                    continue
                parts.append(
                    f"<details><summary>{html.escape(rollout['name'])} — trajectory</summary>"
                    f"<pre>{html.escape(rollout['trajectory'])}</pre></details>"
                )
            if task["selection"]:
                rows = "".join(
                    f"<li><code>{html.escape(c['file'])}</code>"
                    + (f" — total {html.escape(c['total'])}" if c["total"] else "")
                    + "</li>"
                    for c in task["selection"]
                )
                parts.append(f"<p class='muted'>Selection candidates:</p><ul>{rows}</ul>")
    if data["library"]:
        parts.append("<h2>experience library</h2>")
        for entry in data["library"]:
            parts.append(
                f"<details><summary><code>{html.escape(entry['file'])}</code></summary>"
                f"<pre>{html.escape(entry['text'])}</pre></details>"
            )
    parts.append("</main></body></html>")
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument(
        "--run", type=Path, default=Path("runs/demo"), help="Run directory to visualize."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <run>/dashboard.html).",
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the dashboard in the default browser."
    )
    args = parser.parse_args(argv)
    run_root = args.run.expanduser().resolve()
    if not (run_root / "runs").is_dir():
        print(f"error: {run_root} does not look like a run directory (missing runs/)",
              file=sys.stderr)
        return 2
    data = collect(run_root)
    if not data["modes"]:
        print(f"error: no executed tasks found under {run_root}", file=sys.stderr)
        return 2
    output = (args.output or run_root / "dashboard.html").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(data), encoding="utf-8")
    executions = sum(len(t["rollouts"]) for m in data["modes"] for t in m["tasks"])
    print(
        f"dashboard: {output} "
        f"({len(data['modes'])} modes, {executions} executions, "
        f"{len(data['library'])} library entries)"
    )
    if args.open:
        webbrowser.open(output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
