"""Journal -> RESULTS.md + trajectory.png.

    python -m orchestrator.report runs/<run_id>

Reads `journal.jsonl` and nothing else, so it can be run at any moment — mid-run, or after
a crash — and always produces the current state of the deliverable. That is the mitigation
for "the run dies at hour 4 and nobody notices" in the risk register.

`RESULTS.md` is a graded deliverable. It carries the four things the rubric asks for:
the validation-best metrics, the **absolute delta over the official baseline**, the
resource accounting (tokens, wall-clock, iterations used), and the manual-intervention
count that the Autonomy score is read from.

Malformed journal lines are counted and skipped rather than raising: a reporting bug must
never be the reason a finished run has no report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

METRIC_KEYS = ("gauc", "ndcg@5", "primary")

BASELINE_VAL = {"gauc": 0.6674, "ndcg@5": 0.5357, "primary": 0.6016}
BASELINE_TEST = {"gauc": 0.6610, "ndcg@5": 0.5282, "primary": 0.5946}
CEILING = 0.8645
RANDOM_PRIMARY = 0.4753
#: Five-seed std of the official baseline. A gain smaller than this is not a gain;
#: the convergence eps (0.002) is about 2.5x it.
SEED_STD = 0.0008


@dataclass
class RunSummary:
    run_id: str = "unknown"
    model: str = "unknown"
    iterations: int = 0
    scored_nodes: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_s: float = 0.0
    interventions: int = 0
    converged: bool = False
    error_classes: Counter = field(default_factory=Counter)
    recoveries: int = 0
    dead_nodes: int = 0
    best: dict | None = None
    trajectory: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    malformed_lines: int = 0


def read_journal(path: Path) -> list[dict]:
    """Parse journal.jsonl, skipping (and counting) any line that is not valid JSON."""
    events: list[dict] = []
    # Explicit utf-8: Windows defaults to cp1252, and the first em dash or ellipsis an
    # LLM writes into a hypothesis would raise UnicodeDecodeError on read-back. A asked
    # for this directly in STATUS.md.
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                events.append({"event": "__malformed__"})
                continue
            if isinstance(obj, dict):
                events.append(obj)
    return events


def summarise(events: list[dict]) -> RunSummary:
    s = RunSummary()
    for e in events:
        kind = e.get("event")
        if kind == "__malformed__":
            s.malformed_lines += 1
            continue

        s.run_id = e.get("run_id") or s.run_id
        if e.get("model"):
            s.model = e["model"]
        if isinstance(e.get("iteration"), int):
            s.iterations = max(s.iterations, e["iteration"])
        s.tokens_in += int(e.get("tokens_in") or 0)
        s.tokens_out += int(e.get("tokens_out") or 0)
        s.wall_s += float(e.get("wall_s") or 0.0)

        if kind == "proposal":
            s.hypotheses.append(
                {
                    "iteration": e.get("iteration"),
                    "node_id": e.get("node_id"),
                    "kind": e.get("kind"),
                    "hypothesis": (e.get("hypothesis") or "").strip(),
                    "idea_ids": e.get("idea_ids") or [],
                }
            )
        elif kind == "eval":
            m = e.get("metrics") or {}
            if "primary" in m:
                s.scored_nodes += 1
                s.trajectory.append(
                    {
                        "iteration": e.get("iteration"),
                        "node_id": e.get("node_id"),
                        "metrics": m,
                    }
                )
        elif kind == "error":
            s.error_classes[e.get("error_class") or "unknown"] += 1
        elif kind == "recovery":
            s.recoveries += 1
            if (e.get("recovery") or "") == "route_around":
                s.dead_nodes += 1
        elif kind == "intervention":
            s.interventions += 1
        elif kind == "best_updated":
            if e.get("metrics"):
                s.best = {"node_id": e.get("node_id"), "iteration": e.get("iteration"),
                          "metrics": e["metrics"]}
        elif kind == "converged":
            s.converged = True

    # Fall back to the best point on the trajectory if no best_updated event was seen.
    if s.best is None and s.trajectory:
        top = max(s.trajectory, key=lambda t: t["metrics"].get("primary", -1))
        s.best = {"node_id": top["node_id"], "iteration": top["iteration"],
                  "metrics": top["metrics"]}
    return s


def _fmt_delta(v: float) -> str:
    return f"{v:+.4f}"


def _pct_of_headroom(primary: float, ceiling: float = CEILING) -> float:
    """Share of the attainable range (random -> oracle ceiling) captured."""
    span = ceiling - RANDOM_PRIMARY
    return 100.0 * (primary - RANDOM_PRIMARY) / span if span else float("nan")


def render_results(s: RunSummary, facts: dict | None = None) -> str:
    facts = facts or {}
    baseline = facts.get("baseline_val", BASELINE_VAL)
    metric_keys = facts.get("metric_keys", METRIC_KEYS)
    ceiling = facts.get("ceiling", CEILING)
    out: list[str] = []
    A = out.append

    A(f"# Results — `{s.run_id}`")
    A("")
    A("Generated from `journal.jsonl` by `python -m orchestrator.report`.")
    A("")

    A("## Headline")
    A("")
    if s.best:
        m = s.best["metrics"]
        A("| metric | official baseline (val) | our validation best | absolute delta |")
        A("|---|---|---|---|")
        for k in metric_keys:
            if k not in m:
                continue
            if k in baseline:
                A(f"| {k} | {baseline[k]:.4f} | {m[k]:.4f} | "
                  f"**{_fmt_delta(m[k] - baseline[k])}** |")
            else:
                A(f"| {k} | — | {m[k]:.4f} | — |")
        A("")
        line = (
            f"Best node `{s.best['node_id']}` at iteration {s.best['iteration']}. "
            f"Primary {m.get('primary', float('nan')):.4f}."
        )
        # The headroom figure is only meaningful when the task actually declares a
        # ceiling and a baseline. Quoting KuaiRand's for another task would put a wrong
        # number in a graded deliverable.
        if ceiling is not None and "primary" in baseline:
            line += (
                f" That captures {_pct_of_headroom(m.get('primary', 0), ceiling):.1f}% of "
                f"the attainable range (random {RANDOM_PRIMARY} -> ceiling {ceiling}); "
                f"the official baseline captures "
                f"{_pct_of_headroom(baseline['primary'], ceiling):.1f}%."
            )
        A(line)
    else:
        A("No scored node in this journal yet.")
    A("")

    A("## Resource accounting")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| Iterations used | {s.iterations} |")
    A(f"| Scored nodes | {s.scored_nodes} |")
    A(f"| Total LLM tokens (in + out) | **{s.tokens_in + s.tokens_out:,}** |")
    A(f"| — input | {s.tokens_in:,} |")
    A(f"| — output | {s.tokens_out:,} |")
    A(f"| Agent wall-clock | {s.wall_s / 3600:.2f} h ({s.wall_s:,.0f} s) |")
    A("| GPU-hours | 0.00 (CPU only) |")
    A(f"| Model | `{s.model}` |")
    A(f"| Converged | {'yes' if s.converged else 'no'} |")
    A("")

    A("## Autonomy")
    A("")
    A(f"**Manual interventions during this run: {s.interventions}.**")
    A("")
    A("Every human touch is logged as an `intervention` event in the journal and in")
    A("`interventions.md`. Zero is the target; each one is treated as a bug in the agent.")
    A("")

    A("## Robustness")
    A("")
    total_err = sum(s.error_classes.values())
    A(f"- Failed steps: **{total_err}**, of which **{s.recoveries}** were recovered automatically.")
    A(f"- Nodes abandoned after the repair budget, routed around: {s.dead_nodes}.")
    if total_err:
        A("")
        A("| error class | count |")
        A("|---|---|")
        for cls, n in s.error_classes.most_common():
            A(f"| {cls} | {n} |")
    A("")
    A("A failed node never stops the run: it is repaired up to three times, then marked dead")
    A("and the search routes around it.")
    A("")

    A("## Score trajectory")
    A("")
    if s.trajectory:
        A("| iter | node | GAUC | nDCG@5 | primary | vs baseline |")
        A("|---|---|---|---|---|---|")
        for t in s.trajectory:
            m = t["metrics"]
            A(
                f"| {t['iteration']} | `{t['node_id']}` | {m.get('gauc', float('nan')):.4f} | "
                f"{m.get('ndcg@5', float('nan')):.4f} | {m.get('primary', float('nan')):.4f} | "
                f"{_fmt_delta(m.get('primary', 0) - BASELINE_VAL['primary'])} |"
            )
    else:
        A("_No scored iterations yet._")
    A("")

    A("## What the agent tried, and why")
    A("")
    A("The hypothesis behind every proposal, in order. This is the Innovation record.")
    A("")
    if s.hypotheses:
        for h in s.hypotheses:
            ideas = f" _(ideas: {', '.join(h['idea_ids'])})_" if h["idea_ids"] else ""
            A(f"**{h['iteration']} · `{h['node_id']}` · {h['kind']}**{ideas}")
            A("")
            A(f"> {h['hypothesis'] or '_(none recorded)_'}")
            A("")
    else:
        A("_No proposals recorded yet._")

    if s.malformed_lines:
        A("")
        A(f"> {s.malformed_lines} journal line(s) could not be parsed and were skipped.")

    return "\n".join(out) + "\n"


def write_trajectory_png(s: RunSummary, path: Path, facts: dict | None = None) -> bool:
    """Plot validation primary per iteration against the baseline. Optional dependency."""
    if not s.trajectory:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    xs = [t["iteration"] for t in s.trajectory]
    ys = [t["metrics"].get("primary") for t in s.trajectory]
    running = []
    best = float("-inf")
    for y in ys:
        best = max(best, y)
        running.append(best)

    facts = facts or {}
    baseline = facts.get("baseline_val", BASELINE_VAL)
    ceiling = facts.get("ceiling", CEILING)
    seed_std = facts.get("seed_std") or SEED_STD
    base = baseline.get("primary")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)

    # Reference lines only when the task declares them. On KuaiRand the ceiling is the
    # point of the chart — a perfect ranking reaches 0.8645, not 1.0 — but a task with no
    # published baseline must not be drawn against someone else's.
    if ceiling is not None:
        ax.axhline(ceiling, ls=":", lw=1.4, color="#2f6fd0",
                   label=f"attainable ceiling {ceiling:.4f}")
    if base is not None:
        # +/- one seed std around the baseline. Inside this band is noise, not a win.
        ax.axhspan(base - seed_std, base + seed_std, color="crimson", alpha=0.13,
                   label=f"seed noise ±{seed_std:.4f}")
        ax.axhline(base, ls="--", lw=1.4, color="crimson",
                   label=f"official baseline {base:.4f}")

    ax.plot(xs, ys, "o-", ms=4, lw=1, alpha=0.55, label="node validation primary")
    ax.plot(xs, running, lw=2.2, label="best so far")

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if len(xs) == 1:  # a lone point otherwise renders on a fractional axis
        ax.set_xlim(xs[0] - 1, xs[0] + 1)
    refs = [v for v in (base, ceiling) if v is not None]
    lo = min([*ys, *refs]) - 0.01 * max(1.0, abs(min([*ys, *refs])))
    hi = max([*ys, *refs]) + 0.015 * max(1.0, abs(max([*ys, *refs])))
    ax.set_ylim(lo, hi)
    ax.set_xlabel("iteration")
    ax.set_ylabel("validation primary")
    ax.set_title(f"Score trajectory — {s.run_id}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def task_facts(run_dir: Path) -> dict:
    """Baseline, ceiling and metric names for the run, read from its own `config.json`.

    The module constants below are KuaiRand's. Reporting them for another task would put
    someone else's baseline in a results table — a wrong number in a graded deliverable,
    which is worse than no number. Falls back to the constants when there is no config.
    """
    facts = {
        "baseline_val": dict(BASELINE_VAL),
        "ceiling": CEILING,
        "seed_std": SEED_STD,
        "metric_keys": METRIC_KEYS,
        "name": None,
    }
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        return facts
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return facts
    task = cfg.get("task") or {}
    if task.get("name"):
        facts["name"] = task["name"]
    if isinstance(task.get("baseline_val"), dict):
        facts["baseline_val"] = {k: float(v) for k, v in task["baseline_val"].items()}
    if "ceiling" in task:
        facts["ceiling"] = task["ceiling"]
    metrics = task.get("metrics")
    if metrics:
        facts["metric_keys"] = (*metrics, "primary")
    return facts


def build(run_dir: Path) -> RunSummary:
    """Write RESULTS.md (and trajectory.png if matplotlib is available) into `run_dir`."""
    journal = run_dir / "journal.jsonl"
    if not journal.is_file():
        raise FileNotFoundError(f"no journal at {journal}")
    s = summarise(read_journal(journal))
    facts = task_facts(run_dir)
    (run_dir / "RESULTS.md").write_text(render_results(s, facts), encoding="utf-8")
    write_trajectory_png(s, run_dir / "trajectory.png", facts)
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate RESULTS.md from a run journal.")
    # Positional is the documented form. `--run` is accepted because the Makefile's
    # `make report RUN=<id>` target passes it that way; supporting both here is cheaper
    # than a cross-owner change to the Makefile.
    ap.add_argument("run_dir", type=Path, nargs="?", help="runs/<run_id>")
    ap.add_argument("--run", dest="run_flag", type=Path, help="same as run_dir")
    a = ap.parse_args(argv)
    run_dir = a.run_dir or a.run_flag
    if run_dir is None:
        ap.error("give a run directory, either positionally or with --run")
    if not run_dir.exists() and (Path("runs") / str(run_dir)).exists():
        run_dir = Path("runs") / str(run_dir)  # `make report RUN=r2026...` passes a bare id
    s = build(run_dir)
    best = s.best["metrics"]["primary"] if s.best else float("nan")
    print(
        f"wrote {run_dir / 'RESULTS.md'} — {s.scored_nodes} scored nodes, "
        f"best primary {best:.4f}, {s.interventions} interventions"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
