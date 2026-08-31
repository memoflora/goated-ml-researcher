"""CLI entry point. Owner: A.

    python -m orchestrator.run --task kuairand-pure --mode {smoke,dev,official} \
           [--max-iters N] [--wall-clock 6h] [--resume RUN_ID] [--seed N]

Fully unattended by default. There is no interactive mode, and there will not
be one: wanting to answer a question mid-run is a signal to make the agent
handle the case instead. Autonomy is directly scored.

Components are resolved per seam. `auto` uses the real module if it imports and
the stub if it does not, so nobody is ever blocked on someone else's file.
`smoke` pins all three to stubs: `make check` must stay green and under 60 s no
matter what has landed, and it must not need the dataset.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from orchestrator import journal as journal_mod
from orchestrator.contracts import TaskSpec
from orchestrator.core import MODES, Orchestrator, new_run_id, task_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINE_VAL = {"gauc": 0.6674, "ndcg@5": 0.5357, "primary": 0.6016}
BASELINE_TEST = {"gauc": 0.6610, "ndcg@5": 0.5282, "primary": 0.5946}


def parse_duration(text: str) -> int:
    """'6h' | '90m' | '3600s' | '3600' -> seconds."""
    text = str(text).strip().lower()
    units = {"h": 3600, "m": 60, "s": 1}
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(float(text))


def build_task(args: argparse.Namespace, cfg: dict) -> TaskSpec:
    """Build the run's `TaskSpec`, from `tasks/<name>.yaml` when one exists.

    Falling back to the hardcoded KuaiRand constants keeps every old invocation and every
    stubbed test working even with no task file present — `make check` must stay green
    without a dataset on disk.
    """
    # Precedence: an explicit flag wins, then the mode, then the task file.
    #
    # The mode has to beat the task here. `smoke` and `dev` exist precisely to be small
    # and fast whatever the task claims it needs — letting a task's `max_iters: 50` leak
    # into a smoke run turns `make check` into a 50-iteration job. Only `official` takes
    # its ceiling from the task, which is where a per-task budget actually belongs.
    max_iters = args.max_iters if args.max_iters is not None else cfg["max_iters"]
    wall_clock_s = (
        parse_duration(args.wall_clock) if args.wall_clock is not None else cfg["wall_clock_s"]
    )
    task_sets_limits = args.mode == "official" and args.max_iters is None

    tc = None
    try:
        from orchestrator.taskspec import TaskConfigError, load_task

        tc = load_task(args.task)
    except TaskConfigError as exc:
        # A *malformed* task file is a real error and must not be silently ignored; a
        # simply absent one is fine, that is the legacy path.
        if "no task file at" not in str(exc):
            raise SystemExit(f"task config error: {exc}") from None
    except ImportError:
        pass

    if tc is None:
        return TaskSpec(
            name=args.task,
            data_dir=Path(args.data_dir),
            metrics=("gauc", "ndcg@5"),
            baseline_val=dict(BASELINE_VAL),
            baseline_test=dict(BASELINE_TEST),
            max_iters=max_iters,
            wall_clock_s=wall_clock_s,
        )

    return TaskSpec(
        name=tc.name,
        data_dir=Path(args.data_dir) if args.data_dir_explicit else tc.data.dir,
        metrics=tc.report_metrics,
        baseline_val=dict(tc.baseline_val),
        baseline_test=dict(tc.baseline_test),
        ceiling=tc.ceiling,
        max_iters=tc.max_iters if task_sets_limits else max_iters,
        wall_clock_s=(
            tc.wall_clock_s if task_sets_limits and args.wall_clock is None else wall_clock_s
        ),
        conv_eps=tc.conv_eps,
        conv_n=tc.conv_n,
        kind=tc.kind,
        description=tc.description,
        primary_parts=tc.primary_parts,
        submission_columns=tc.submission_columns,
        prediction_column=tc.prediction_column,
        seed_std=tc.seed_std,
        config=tc,
    )


def _load(module: str, attrs: tuple[str, ...]) -> Any | None:
    """Import a teammate's module only if it exists and exposes its seam."""
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - a teammate's module not existing yet is normal
        return None
    return mod if all(hasattr(mod, a) for a in attrs) else None


def resolve_components(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, dict]:
    sys.path.insert(0, str(REPO_ROOT))
    from tests.stubs import StubAgent, StubEvaluator, StubExecutor

    chosen: dict[str, str] = {}

    agent = None
    if args.agent == "replay":
        # Canned real pipelines, no key and no tokens. Reachable via TECHJAM_AGENT too,
        # but a flag is what people actually find: this is the path that exercises the
        # whole loop for free, so it should not be discoverable only by reading code.
        agent_mod = _load("orchestrator.agent", ("draft", "improve", "repair"))
        if agent_mod is not None:
            agent, chosen["agent"] = agent_mod.ReplayAgent(), "replay"
    elif args.agent == "auto":
        agent = _load("orchestrator.agent", ("draft", "improve", "repair"))
        if agent is not None:
            chosen["agent"] = "orchestrator.agent"
    if agent is None:
        agent, chosen["agent"] = StubAgent(), "stub"

    executor = None
    if args.sandbox == "auto":
        executor = _load("orchestrator.sandbox", ("run",))
    if executor is None:
        executor, chosen["sandbox"] = StubExecutor(), "stub"
    else:
        chosen["sandbox"] = "orchestrator.sandbox"

    evaluator = None
    if args.evaluator == "auto":
        evaluator = _load("orchestrator.evaluate", ("score", "validate"))
    if evaluator is None:
        evaluator, chosen["evaluator"] = StubEvaluator(), "stub"
    else:
        chosen["evaluator"] = "orchestrator.evaluate"

    knowledge = _load("orchestrator.knowledge", ("retrieve",))
    chosen["knowledge"] = "orchestrator.knowledge" if knowledge else "none"
    datacard = _load("orchestrator.datacard", ("data_card",))
    chosen["datacard"] = "orchestrator.datacard" if datacard else "none"

    return agent, executor, evaluator, knowledge, datacard, chosen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m orchestrator.run",
        description=(
            "Run the autonomous ML research agent on a task. --task names a file in "
            "tasks/ (or a path to one); see tasks/kuairand-pure.yaml for the shape."
        ),
    )
    ap.add_argument(
        "--task",
        default="kuairand-pure",
        help="task name in tasks/, or a path to a task YAML file",
    )
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="print the tasks defined in tasks/ and exit",
    )
    ap.add_argument("--mode", choices=sorted(MODES), default="smoke")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--wall-clock", default=None, help="e.g. 6h, 90m, 3600s")
    ap.add_argument("--resume", metavar="RUN_ID", default=None)
    ap.add_argument("--seed", type=int, default=0)
    # Default is None so we can tell "user chose a directory" from "use the task's".
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--subsample", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=None, help="per-pipeline timeout, seconds")
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--agent", choices=("auto", "stub", "replay"), default=None)
    ap.add_argument("--sandbox", choices=("auto", "stub"), default=None)
    ap.add_argument("--evaluator", choices=("auto", "stub"), default=None)
    args = ap.parse_args(argv)
    args.data_dir_explicit = args.data_dir is not None
    if args.data_dir is None:
        args.data_dir = "data/kuairand-pure"
    # smoke pins every seam to a stub; every other mode prefers the real module.
    default_source = "stub" if args.mode == "smoke" else "auto"
    for seam in ("agent", "sandbox", "evaluator"):
        if getattr(args, seam) is None:
            setattr(args, seam, default_source)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_tasks:
        from orchestrator.taskspec import TASKS_DIR, available_tasks, load_task

        names = available_tasks()
        if not names:
            print(f"no task files in {TASKS_DIR}")
            return 1
        for name in names:
            try:
                t = load_task(name)
                first = (t.description.strip().splitlines() or [""])[0]
                print(f"{name:24} {t.kind:12} primary={','.join(t.primary_parts):18} {first[:60]}")
            except Exception as exc:  # noqa: BLE001 - report, do not crash the listing
                print(f"{name:24} <invalid: {exc}>")
        return 0

    cfg = MODES[args.mode]
    run_id = args.resume or new_run_id(runs_dir=args.runs_dir)
    run_dir = Path(args.runs_dir) / run_id
    agent, executor, evaluator, knowledge, datacard, chosen = resolve_components(args)

    common: dict[str, Any] = {
        "agent": agent,
        "executor": executor,
        "evaluator": evaluator,
        "knowledge": knowledge,
        "datacard": datacard,
        "mode": args.mode,
        "seed": args.seed,
        "subsample": args.subsample,
        "timeout_s": args.timeout,
        "token_budget": args.token_budget,
        "components": chosen,
    }

    if args.resume:
        if not (run_dir / "state.json").exists():
            print(f"no checkpoint at {run_dir/'state.json'}", file=sys.stderr)
            return 2
        # Resume keeps the original run's task, but --max-iters / --wall-clock
        # still override it: a run that stopped on a ceiling must be extendable
        # without a human editing config.json.
        task = task_from_config(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        overrides = {}
        if args.max_iters is not None:
            overrides["max_iters"] = args.max_iters
        if args.wall_clock is not None:
            overrides["wall_clock_s"] = parse_duration(args.wall_clock)
        if overrides:
            task = replace(task, **overrides)
        orch = Orchestrator.resume(run_dir, run_id, task=task, **common)
    else:
        orch = Orchestrator(build_task(args, cfg), run_dir=run_dir, run_id=run_id, **common)

    summary = orch.run()
    journal_mod.close()

    print(json.dumps(summary, indent=1))
    print(f"\njournal: {run_dir/'journal.jsonl'}")
    print(f"final:   {summary.get('final_submission')}")
    return 0 if summary.get("best_node") else 1


if __name__ == "__main__":
    raise SystemExit(main())
