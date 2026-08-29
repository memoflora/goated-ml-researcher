"""A TaskSpec that needs no dataset on disk."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import TaskSpec

BASELINE_VAL = {"gauc": 0.6674, "ndcg@5": 0.5357, "primary": 0.6016}
BASELINE_TEST = {"gauc": 0.6610, "ndcg@5": 0.5282, "primary": 0.5946}


def stub_task(
    data_dir: Path | str = "data/kuairand-pure",
    *,
    max_iters: int = 3,
    wall_clock_s: int = 300,
) -> TaskSpec:
    return TaskSpec(
        name="kuairand-pure",
        data_dir=Path(data_dir),
        metrics=("gauc", "ndcg@5"),
        baseline_val=dict(BASELINE_VAL),
        baseline_test=dict(BASELINE_TEST),
        max_iters=max_iters,
        wall_clock_s=wall_clock_s,
    )
