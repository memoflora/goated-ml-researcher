"""Frozen dataclasses shared by every module.

OWNER: A (ML Engineer — Orchestrator & Run).

B transcribed this file verbatim from `references/contracts.md` §2 at H+0 so that
`sandbox.py` and `agent.py` had something to import. A: overwrite it with yours —
no reconciliation should be needed, the shapes below are a straight copy of the
frozen spec. The one addition is `Context` (+ `HistoryEntry`, `Budget`) at the
bottom, which §3 specifies in prose only; see STATUS.md `## Requests`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

NodeKind = Literal["draft", "improve", "debug"]
NodeStatus = Literal["pending", "running", "ok", "buggy", "timeout", "dead"]
ErrorClass = Literal["syntax", "import", "data", "runtime", "oom",
                     "timeout", "contract", "eval", "unknown"]


@dataclass(frozen=True)
class TaskSpec:
    name: str                       # "kuairand-pure"
    data_dir: Path
    metrics: tuple[str, ...]        # ("gauc", "ndcg@5")
    baseline_val: dict[str, float]  # {"gauc":0.6674,"ndcg@5":0.5357,"primary":0.6016}
    baseline_test: dict[str, float] # {"gauc":0.6610,"ndcg@5":0.5282,"primary":0.5946}
    ceiling: float = 0.8645
    max_iters: int = 50
    wall_clock_s: int = 6 * 3600
    conv_eps: float = 0.002
    conv_n: int = 3


@dataclass
class Idea:                         # D produces
    id: str                         # "T2.lgbm-on-engineered"
    tier: int                       # 0..4, increasing effort/payoff
    title: str
    summary: str                    # 2-4 sentences the LLM can act on
    citation: str | None
    est_minutes: int
    prerequisites: list[str]


@dataclass
class Proposal:                     # B produces, A consumes
    hypothesis: str                 # WHY, one paragraph. Scored under Innovation.
    plan: list[str]                 # WHAT changes, 3-6 bullets
    code: str                       # the full new pipeline.py
    idea_ids: list[str]             # ideas drawn on, may be empty
    tokens_in: int
    tokens_out: int
    model: str


@dataclass
class ExecResult:                   # B produces
    ok: bool
    exit_code: int
    stdout_tail: str                # last 4000 chars
    stderr_tail: str                # last 4000 chars
    error_class: ErrorClass | None
    error_excerpt: str | None       # most useful traceback slice, <= 1500 chars
    result_json: dict | None        # parsed from the RESULT_JSON line
    artifacts: dict[str, Path]      # {"submission": Path(...)}
    wall_s: float
    peak_rss_mb: float


@dataclass
class Node:
    id: str                         # "n007"
    parent_id: str | None
    kind: NodeKind
    iteration: int
    workspace: Path                 # runs/<run_id>/nodes/n007/
    proposal: Proposal | None = None
    exec_result: ExecResult | None = None
    metrics: dict[str, float] | None = None
    status: NodeStatus = "pending"
    repair_attempts: int = 0
    children: list[str] = field(default_factory=list)


# --- proposed by B, awaiting A's ack (contracts.md §3 specifies these in prose) ---

@dataclass
class HistoryEntry:
    """One past iteration, compacted. Hypothesis + metric delta only, never code."""
    iteration: int
    kind: NodeKind
    hypothesis: str
    status: NodeStatus
    primary: float | None = None
    delta_primary: float | None = None
    error_class: ErrorClass | None = None


@dataclass
class Budget:
    iters_left: int
    seconds_left: float
    tokens_left: int


@dataclass
class Context:
    """Assembled by A, consumed by B's agent. Everything the LLM is allowed to see."""
    task: TaskSpec
    data_card: str                              # C's markdown EDA summary
    ideas: list[Idea] = field(default_factory=list)          # D's top-K
    history: list[HistoryEntry] = field(default_factory=list)
    budget: Budget | None = None
    parent_code: str | None = None
    parent_metrics: dict[str, float] | None = None
    library_whitelist: tuple[str, ...] = ()     # from requirements-pipeline.txt (C)
    run_id: str = ""
    iteration: int = 0
    draft_angle: str | None = None              # varies the three draft-phase calls
