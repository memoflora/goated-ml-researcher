"""Frozen dataclasses shared by every module. Owner: A.

Frozen after H+2. To change anything here, add a `## Contract change proposed`
entry to STATUS.md, keep the old shape working, and get one ack.

Nothing in this file may import from any other module in `orchestrator/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

NodeKind = Literal["draft", "improve", "debug"]
NodeStatus = Literal["pending", "running", "ok", "buggy", "timeout", "dead"]
ErrorClass = Literal[
    "syntax", "import", "data", "runtime", "oom", "timeout", "contract", "eval", "unknown"
]

#: Journal `event` values. The journal is a graded deliverable; keep this closed.
EventKind = Literal[
    "run_start",
    "data_card",
    "proposal",
    "exec",
    "eval",
    "error",
    "recovery",
    "prune",
    "best_updated",
    "intervention",
    "converged",
    "run_end",
]

#: The scored metric is the equal-weighted mean of these two.
PRIMARY_PARTS: tuple[str, str] = ("gauc", "ndcg@5")


def primary(metrics: dict[str, float]) -> float:
    """Equal-weighted mean of GAUC and nDCG@5.

    `evaluate.score()` (C) already returns `primary`; this is the fallback for
    stubs and for journal rows written before C's evaluator was wired in.
    """
    if "primary" in metrics:
        return float(metrics["primary"])
    return sum(float(metrics[k]) for k in PRIMARY_PARTS) / len(PRIMARY_PARTS)


@dataclass(frozen=True)
class TaskSpec:
    name: str  # "kuairand-pure"
    data_dir: Path
    metrics: tuple[str, ...]  # ("gauc", "ndcg@5")
    baseline_val: dict[str, float]  # {"gauc":0.6674,"ndcg@5":0.5357,"primary":0.6016}
    baseline_test: dict[str, float]  # {"gauc":0.6610,"ndcg@5":0.5282,"primary":0.5946}
    ceiling: float = 0.8645
    max_iters: int = 50
    wall_clock_s: int = 6 * 3600
    conv_eps: float = 0.002
    conv_n: int = 3


@dataclass
class Idea:  # D produces
    id: str  # "T2.lgbm-on-engineered"
    tier: int  # 0..4, increasing effort/payoff
    title: str
    summary: str  # 2-4 sentences the LLM can act on
    citation: str | None
    est_minutes: int
    prerequisites: list[str]


@dataclass
class Proposal:  # B produces, A consumes
    hypothesis: str  # WHY, one paragraph. Scored under Innovation.
    plan: list[str]  # WHAT changes, 3-6 bullets
    code: str  # the full new pipeline.py
    idea_ids: list[str]  # ideas drawn on, may be empty
    tokens_in: int
    tokens_out: int
    model: str


@dataclass
class ExecResult:  # B produces
    ok: bool
    exit_code: int
    stdout_tail: str  # last 4000 chars
    stderr_tail: str  # last 4000 chars
    error_class: ErrorClass | None
    error_excerpt: str | None  # most useful traceback slice, <= 1500 chars
    result_json: dict | None  # parsed from the RESULT_JSON line
    artifacts: dict[str, Path]  # {"submission": Path(...)}
    wall_s: float
    peak_rss_mb: float


@dataclass
class Node:
    id: str  # "n007"
    parent_id: str | None
    kind: NodeKind
    iteration: int
    workspace: Path  # runs/<run_id>/nodes/n007/
    proposal: Proposal | None = None
    exec_result: ExecResult | None = None
    metrics: dict[str, float] | None = None
    status: NodeStatus = "pending"
    repair_attempts: int = 0
    children: list[str] = field(default_factory=list)

    @property
    def primary(self) -> float | None:
        """Validation primary, or None if this node was never scored."""
        return primary(self.metrics) if self.metrics else None


# --------------------------------------------------------------------------
# Context — assembled by A, consumed by B's agent.py. Part of the frozen seam.
# --------------------------------------------------------------------------


@dataclass
class HistoryEntry:
    """One past iteration, compacted. Never carries code — that is the whole point."""

    iteration: int
    node_id: str
    kind: NodeKind
    hypothesis: str  # may be truncated by A before it goes in
    primary: float | None  # None when the node errored
    delta_vs_baseline: float | None
    status: NodeStatus
    error_class: ErrorClass | None = None


@dataclass
class Budget:
    """What is left. B uses this to size its prompts; A refuses to overrun it."""

    iters_left: int
    seconds_left: float
    tokens_left: int
    tokens_in_used: int = 0
    tokens_out_used: int = 0


@dataclass
class Context:
    """Everything the agent is allowed to see when it writes the next pipeline.

    Assembled by A (`core.build_context`), passed to B's `draft/improve/repair`.
    Deliberately small: no dataset, no full history, at most one parent program.
    """

    task: TaskSpec
    run_id: str
    iteration: int
    data_card: str  # C's markdown EDA summary, <= 3000 tokens
    ideas: list[Idea]  # D's top-K
    history: list[HistoryEntry]  # compact, most recent last
    budget: Budget
    library_whitelist: list[str]  # contents of requirements-pipeline.txt
    pipeline_cli: str  # the frozen `python pipeline.py ...` contract, verbatim
    baseline_val: dict[str, float]

    # improve / repair only — None on a draft
    parent_code: str | None = None
    parent_metrics: dict[str, float] | None = None
    parent_hypothesis: str | None = None

    # repair only
    error_class: ErrorClass | None = None
    error_excerpt: str | None = None
    stderr_tail: str | None = None
    prior_repair_plans: list[str] = field(default_factory=list)
    repair_attempt: int = 0

    # draft only — A varies this so three drafts are not three identical drafts
    draft_angle: str | None = None


#: The pipeline CLI contract, verbatim from contracts.md §1. B puts this in the
#: prompt; the sandbox invokes exactly this shape. Single source of truth.
PIPELINE_CLI = (
    "python pipeline.py --data-dir DIR --out-dir DIR --split {val,test} "
    "--seed N [--subsample F]"
)
