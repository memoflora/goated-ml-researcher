"""Per-iteration run log — the Starter Kit's Run-log requirement.

    python -m orchestrator.runlog runs/<run_id>

Writes `RUNLOG.md` next to the journal. The problem statement (§2.5 item 3) asks for
four things per iteration, and they come from three different places:

    hypothesis          journal `proposal.hypothesis` — what it meant to try, and why
    the code diff       reconstructed here, from the node workspaces
    resulting metrics   journal `eval.metrics` — GAUC / nDCG@5 and delta vs baseline
    errors and recovery journal `error` and `recovery` — and how the agent handled it

plus a manual-intervention count, which is `interventions.md`.

**The diff is the only part no existing artifact carries.** That is deliberate rather
than an oversight: the journal never stores code, because carrying past code into
prompts is how a token budget dies (`core.build_context` says so explicitly). Each node
keeps its own `pipeline.py` on disk, so the diff against its parent is reconstructed
from those two files instead of being logged fifty times over.

Reads the run directory and nothing else, so it can be regenerated at any point —
including mid-run, or from a run that crashed before writing a summary.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

#: Long diffs make the log unreadable and are already on disk in full. Truncate with a
#: pointer rather than dropping them silently.
DEFAULT_MAX_DIFF_LINES = 120


def read_journal(path: Path) -> list[dict]:
    """Tolerates a torn last line: a run killed mid-write still logs everything before."""
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def load_tree(run_dir: Path) -> dict[str, dict]:
    """Node id -> node, from the checkpoint. Absent on a run killed before iteration 1."""
    state = run_dir / "state.json"
    if not state.is_file():
        return {}
    try:
        return json.loads(state.read_text(encoding="utf-8")).get("tree", {}).get("nodes", {})
    except (json.JSONDecodeError, AttributeError):
        return {}


def pipeline_of(run_dir: Path, node_id: str | None) -> list[str]:
    if not node_id:
        return []
    src = run_dir / "nodes" / node_id / "pipeline.py"
    if not src.is_file():
        return []
    return src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)


def diff_lines(run_dir: Path, node_id: str, parent_id: str | None) -> tuple[list[str], str]:
    """Unified diff of a node's pipeline against its parent's.

    Returns (lines, note). A draft has no parent, so there is nothing to diff against —
    say so and give the size, rather than dumping a few hundred lines of new file that
    are already in the workspace.
    """
    new = pipeline_of(run_dir, node_id)
    if not new:
        return [], "_no `pipeline.py` on disk for this node_"
    old = pipeline_of(run_dir, parent_id)
    if not old:
        return [], (
            f"_new program (no parent to diff against) — {len(new)} lines, "
            f"full source at `nodes/{node_id}/pipeline.py`_"
        )
    lines = list(difflib.unified_diff(
        old, new,
        fromfile=f"nodes/{parent_id}/pipeline.py",
        tofile=f"nodes/{node_id}/pipeline.py",
        n=3,
    ))
    if not lines:
        return [], "_no change to `pipeline.py`_"
    return lines, ""


def count_interventions(run_dir: Path) -> tuple[int, str]:
    """Data rows in the `interventions.md` table. Autonomy is scored directly on this.

    `interventions.md` is a markdown table with a prose preamble, so counting
    non-empty lines reports 4 for an empty log — the two sentences of preamble plus
    the header and separator. Count table *body* rows and nothing else: putting a
    wrong number on a scored deliverable is worse than putting none.
    """
    path = run_dir / "interventions.md"
    if not path.is_file():
        return 0, "no `interventions.md` in the run directory"
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue                                   # |---|---| separator
        if [c.lower() for c in cells[:2]] == ["utc", "who"]:
            continue                                   # header
        if any(cells):
            rows += 1
    return rows, ""


def render(run_dir: Path, *, max_diff_lines: int = DEFAULT_MAX_DIFF_LINES) -> str:
    run_dir = Path(run_dir)
    events = read_journal(run_dir / "journal.jsonl")
    tree = load_tree(run_dir)

    start = next((e for e in events if e.get("event") == "run_start"), {})
    out: list[str] = []
    A = out.append

    A(f"# Run log — `{start.get('run_id', run_dir.name)}`")
    A("")
    A("Per-iteration record required by the Starter Kit: the hypothesis the agent formed,")
    A("the code diff it applied, the metrics that came back, and any error it recovered")
    A("from. Generated from the run directory by `python -m orchestrator.runlog`.")
    A("")
    A(f"- task: `{start.get('task', '?')}` · mode: `{start.get('mode', '?')}` "
      f"· model: `{_model_of(events)}`")
    A(f"- commit: `{start.get('git_sha', '?')}` · subsample: `{start.get('subsample')}`")
    A(f"- limits: {start.get('max_iters', '?')} iterations, "
      f"{start.get('wall_clock_s', '?')}s wall-clock, "
      f"conv_eps={start.get('conv_eps')} conv_n={start.get('conv_n')} "
      f"explore_after={start.get('explore_after')}")
    A("")

    n_int, note = count_interventions(run_dir)
    A("## Manual interventions")
    A("")
    A(f"**{n_int}**" + (f" — {note}" if note else
      " — every iteration below ran unattended, from launch to the final summary."))
    A("")

    by_iter: dict[int, list[dict]] = {}
    for e in events:
        if e.get("event") in ("proposal", "eval", "error", "recovery", "exec"):
            by_iter.setdefault(int(e.get("iteration") or 0), []).append(e)

    A("## Iterations")
    A("")
    if not by_iter:
        A("_No iterations recorded._")
        return "\n".join(out) + "\n"

    for it in sorted(by_iter):
        evs = by_iter[it]
        prop = next((e for e in evs if e["event"] == "proposal"), None)
        ev = next((e for e in evs if e["event"] == "eval"), None)
        errs = [e for e in evs if e["event"] == "error"]
        recs = [e for e in evs if e["event"] == "recovery"]
        node_id = (prop or {}).get("node_id") or (ev or {}).get("node_id") or "?"
        parent_id = (tree.get(node_id) or {}).get("parent_id")

        A(f"### Iteration {it} — `{node_id}`"
          + (f" ({prop['kind']} from `{parent_id}`)" if prop and parent_id
             else f" ({prop['kind']})" if prop else ""))
        A("")

        A("**Hypothesis** — what it intended to try, and why")
        A("")
        A(f"> {(prop or {}).get('hypothesis') or '_(none recorded)_'}")
        A("")
        plan = (prop or {}).get("plan")
        if plan:
            A("**Plan**")
            A("")
            for step in (plan if isinstance(plan, list) else [plan]):
                A(f"- {step}")
            A("")
        if (prop or {}).get("idea_ids"):
            A(f"**Ideas drawn on:** {', '.join(prop['idea_ids'])}")
            A("")

        A("**Code diff applied**")
        A("")
        lines, note = diff_lines(run_dir, node_id, parent_id)
        if note:
            A(note)
        else:
            shown = lines[:max_diff_lines]
            A("```diff")
            for ln in shown:
                A(ln.rstrip("\n"))
            A("```")
            if len(lines) > max_diff_lines:
                A(f"_diff truncated at {max_diff_lines} of {len(lines)} lines; "
                  f"full source at `nodes/{node_id}/pipeline.py`_")
        A("")

        A("**Resulting metrics**")
        A("")
        if ev and ev.get("metrics"):
            m, d = ev["metrics"], ev.get("delta_vs_baseline") or {}
            A("| metric | value | vs baseline |")
            A("|---|---|---|")
            for k in sorted(m):
                dv = d.get(k)
                A(f"| {k} | {m[k]:.5f} | {dv:+.5f} |" if isinstance(dv, int | float)
                  else f"| {k} | {m[k]:.5f} | — |")
        else:
            A("_no metrics — this iteration did not produce a scored submission._")
        A("")

        if errs or recs:
            A("**Errors and recovery**")
            A("")
            for e in errs:
                A(f"- `error` · class `{e.get('error_class')}` — "
                  f"{(e.get('error_excerpt') or '').strip()[:400] or '_(no excerpt)_'}")
            for r in recs:
                A(f"- `recovery` · **{r.get('recovery')}**"
                  + (f" (repair {r.get('repair_attempt')}/{r.get('max_repairs')})"
                     if r.get("repair_attempt") else ""))
            A("")

    return "\n".join(out) + "\n"


def _model_of(events: list[dict]) -> str:
    for e in events:
        if e.get("event") == "proposal" and e.get("model"):
            return str(e["model"])
    return "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--max-diff-lines", type=int, default=DEFAULT_MAX_DIFF_LINES)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="default: <run_dir>/RUNLOG.md")
    args = ap.parse_args(argv)

    if not (args.run_dir / "journal.jsonl").is_file():
        raise SystemExit(f"no journal.jsonl in {args.run_dir}")
    text = render(args.run_dir, max_diff_lines=args.max_diff_lines)
    dest = args.out or (args.run_dir / "RUNLOG.md")
    dest.write_text(text, encoding="utf-8")
    print(f"wrote {dest} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
