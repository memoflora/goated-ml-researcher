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


def primary(metrics: dict[str, float], parts: tuple[str, ...] | None = None) -> float:
    """The composite score, always oriented so that **higher is better**.

    `evaluate.score()` (C) already returns `primary` for every task, so this is mostly a
    fallback for stubs and for journal rows written before the evaluator was wired in.
    `parts` defaults to KuaiRand's pair; pass a task's own parts for anything else. A
    lower-is-better member (RMSE, log loss) is negated, so the orchestrator's `>` never
    has to know which direction a metric runs in.
    """
    if "primary" in metrics:
        return float(metrics["primary"])
    parts = parts or PRIMARY_PARTS
    try:
        from orchestrator import metrics as _M

        return _M.primary_of(metrics, parts)
    except (ImportError, KeyError):
        # Unknown metric names (stub fixtures) fall back to a plain mean.
        return sum(float(metrics[k]) for k in parts) / len(parts)


@dataclass(frozen=True)
class TaskSpec:
    """What the run is working on.

    Everything after `conv_n` was added when the orchestrator stopped assuming KuaiRand.
    All of it has a KuaiRand-shaped default, so a `TaskSpec` built the old way still
    behaves exactly as it did, and `tasks/*.yaml` fills the rest in for any other problem.
    """

    name: str  # "kuairand-pure"
    data_dir: Path
    metrics: tuple[str, ...]  # ("gauc", "ndcg@5")
    baseline_val: dict[str, float]  # {"gauc":0.6674,"ndcg@5":0.5357,"primary":0.6016}
    baseline_test: dict[str, float]  # {"gauc":0.6610,"ndcg@5":0.5282,"primary":0.5946}
    ceiling: float | None = 0.8645
    max_iters: int = 50
    wall_clock_s: int = 6 * 3600
    conv_eps: float = 0.002
    conv_n: int = 3
    #: Flat scored iterations before the policy stops improving the best node and
    #: explores the second-best distinct one instead. **Must be < `conv_n`.**
    #: `best_history` is monotone, so `converged()` can only fire once the last
    #: `conv_n` iterations were each flat — meaning `flat_iters >= conv_n` by then.
    #: If this is not strictly smaller, the run always stops on the very iteration
    #: exploration became reachable and the explore branch never executes at all.
    explore_after: int = 2

    #: "ranking" | "binary" | "multiclass" | "regression"
    kind: str = "ranking"
    #: The problem statement, in the user's words. Goes into every prompt.
    description: str = ""
    #: Metrics whose (sign-corrected) mean is `primary`. Always maximised.
    primary_parts: tuple[str, ...] = PRIMARY_PARTS
    submission_columns: tuple[str, ...] = ("row_id", "user_id", "video_id", "score")
    prediction_column: str = "score"
    #: Run-to-run noise of a fixed pipeline. A gain smaller than this is not a gain.
    seed_std: float | None = 0.0008
    #: The parsed `tasks/<name>.yaml`, when the run was started from one.
    config: object | None = None

    def primary(self, metrics: dict[str, float]) -> float:
        """This task's composite score, oriented so higher is always better."""
        if "primary" in metrics:
            return float(metrics["primary"])
        return primary(metrics, self.primary_parts)


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
