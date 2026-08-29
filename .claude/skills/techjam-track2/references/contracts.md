# Frozen contracts

These are the seams that let four people work in parallel. **Role A owns `orchestrator/contracts.py`
and writes these first (by H+2). After that they are frozen.** To change one: add a
`## Contract change proposed` entry in `STATUS.md`, keep the old shape working, and get one
other person to ack before merging.

Everything below is the *shape*, not the implementation. Code against the shape from hour zero;
stub whatever does not exist yet.

---

## 1. The pipeline contract — the single most important interface

Every solution the agent produces is **one self-contained file, `pipeline.py`**, written into
that node's workspace. Nothing else. No packages, no plugin registry, no imports from our repo.

```
python pipeline.py --data-dir DIR --out-dir DIR --split {val,test} --seed N [--subsample F]
```

It must:

1. read the fixed splits from `--data-dir`
2. train using **train only** when `--split val`; train using **train + validation** when `--split test`
3. write `<out-dir>/submission.csv` in the starter-kit schema (`row_id,user_id,video_id,score`)
4. print exactly one line to stdout of the form `RESULT_JSON {...}` containing at minimum
   `{"n_rows": int, "train_seconds": float, "notes": str}`
5. exit 0 on success, non-zero on failure, and never prompt for input
6. honour `--subsample F` (float in (0,1]) by sampling *users*, not rows, for smoke runs

Why one file: trivial to sandbox, trivial to diff, trivial to revert, and the LLM can hold the
whole thing in context. Do not "improve" this into a framework.

Scoring is **never** done inside `pipeline.py`. The orchestrator scores `submission.csv` with
Role C's evaluator. This keeps the agent honest and the metric conventions pinned.

---

## 2. `orchestrator/contracts.py` (owner: A)

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

NodeKind   = Literal["draft", "improve", "debug"]
NodeStatus = Literal["pending", "running", "ok", "buggy", "timeout", "dead"]
ErrorClass = Literal["syntax", "import", "data", "runtime", "oom",
                     "timeout", "contract", "eval", "unknown"]


@dataclass(frozen=True)
class TaskSpec:
    name: str                      # "kuairand-pure"
    data_dir: Path
    metrics: tuple[str, ...]       # ("gauc", "ndcg@5")
    primary_key: str               # "primary"
    baseline_val: dict[str, float] # {"gauc":0.6674,"ndcg@5":0.5357,"primary":0.6016}
    baseline_test: dict[str, float]# {"gauc":0.6610,"ndcg@5":0.5282,"primary":0.5946}
    ceiling: float                 # 0.8645
    max_iters: int = 50
    wall_clock_s: int = 6 * 3600
    conv_eps: float = 0.002
    conv_n: int = 3


@dataclass
class Proposal:                    # what the LLM returns (B produces, A consumes)
    hypothesis: str                # WHY, one paragraph. Feeds the Innovation score.
    plan: str                      # WHAT will change, 3-6 bullets
    code: str                      # the full new pipeline.py
    idea_ids: list[str]            # knowledge-base ideas drawn on, may be empty
    tokens_in: int
    tokens_out: int
    model: str


@dataclass
class ExecResult:                  # B produces
    ok: bool
    exit_code: int
    stdout_tail: str               # last 4000 chars
    stderr_tail: str               # last 4000 chars
    error_class: ErrorClass | None
    error_excerpt: str | None      # the single most useful traceback slice, <= 1500 chars
    result_json: dict | None       # parsed from the RESULT_JSON line
    artifacts: dict[str, Path]     # {"submission": Path(...)}
    wall_s: float
    peak_rss_mb: float


@dataclass
class Node:                        # one solution in the tree
    id: str                        # "n007"
    parent_id: str | None
    kind: NodeKind
    iteration: int
    workspace: Path                # runs/<run_id>/nodes/n007/
    proposal: Proposal | None = None
    exec_result: ExecResult | None = None
    metrics: dict[str, float] | None = None   # {"gauc":..,"ndcg@5":..,"primary":..}
    status: NodeStatus = "pending"
    repair_attempts: int = 0
    children: list[str] = field(default_factory=list)
```

---

## 3. Module interfaces

```python
# orchestrator/exec/  (owner: B)
class Executor(Protocol):
    def run(self, node: Node, *, split: str, seed: int, timeout_s: int,
            subsample: float | None = None) -> ExecResult: ...

# orchestrator/agent/  (owner: B)
class Agent(Protocol):
    def draft(self, ctx: Context) -> Proposal: ...
    def improve(self, ctx: Context, parent: Node) -> Proposal: ...
    def repair(self, ctx: Context, node: Node) -> Proposal: ...

# orchestrator/eval/  (owner: C)
class Evaluator(Protocol):
    def score(self, submission: Path, split: str) -> dict[str, float]: ...
    def validate(self, submission: Path, split: str) -> tuple[bool, str]: ...
    def data_card(self) -> str:      # markdown EDA summary handed to the LLM, <= 3000 tokens
        ...

# orchestrator/knowledge/  (owner: C)
@dataclass(frozen=True)
class Idea:
    id: str                # "T2.lgbm-on-engineered"
    tier: int              # 0..4, roughly increasing effort/payoff
    title: str
    summary: str           # 2-4 sentences the LLM can act on
    citation: str | None
    est_minutes: int
    prerequisites: list[str]

class KnowledgeBase(Protocol):
    def retrieve(self, *, tried: list[str], best_metrics: dict, budget_left: int,
                 k: int = 5) -> list[Idea]: ...

# orchestrator/search/  (owner: A)
class SearchPolicy(Protocol):
    def next_action(self, tree: Tree) -> tuple[NodeKind, Node | None]: ...

# orchestrator/report/  (owner: D)
class Journal(Protocol):
    def emit(self, event: dict) -> None: ...     # appends one JSON line, flushes
```

`Context` is assembled by A and passed to B. It is a plain dataclass holding: `TaskSpec`,
the data card string, the parent node's code + metrics, the top-K `Idea`s, a compact history
of the last few attempts (hypothesis + metric delta only, never full code), and the remaining
iteration/time budget.

---

## 4. Journal schema — `runs/<run_id>/journal.jsonl` (owner: D, emitted by everyone)

One JSON object per line, appended and flushed immediately. This file is a graded deliverable:
judges read it to score Autonomy, Robustness and Innovation. Treat the schema as public API.

```json
{
  "ts": "2026-08-30T04:12:07Z",
  "run_id": "r20260830-0412",
  "iteration": 7,
  "node_id": "n007",
  "parent_id": "n003",
  "event": "proposal",
  "kind": "improve",
  "hypothesis": "Baseline FM ignores user-level exposure frequency; ...",
  "plan": ["add smoothed user CTR prior", "..."],
  "idea_ids": ["T1.user-ctr-prior"],
  "diff_path": "nodes/n007/pipeline.diff",
  "metrics": {"gauc": 0.6791, "ndcg@5": 0.5442, "primary": 0.6117},
  "delta_vs_baseline": {"gauc": 0.0117, "ndcg@5": 0.0085, "primary": 0.0101},
  "error_class": null,
  "recovery": null,
  "tokens_in": 8412,
  "tokens_out": 2190,
  "model": "claude-opus-5",
  "wall_s": 96.4
}
```

`event` is one of:
`run_start`, `data_card`, `proposal`, `exec`, `eval`, `error`, `recovery`, `prune`,
`best_updated`, `intervention`, `converged`, `run_end`.

Fields not relevant to an event are `null` or omitted. Never write anything that is not
valid JSON on a single line. Never log secrets or API keys.

---

## 5. Run directory layout

```
runs/<run_id>/
  config.json          # resolved TaskSpec + CLI args + git sha + model id
  journal.jsonl        # the graded log
  state.json           # tree + budget snapshot, rewritten atomically each iteration
  interventions.md     # every human touch, timestamped. Scored. Keep it empty.
  nodes/n000/pipeline.py, pipeline.diff, stdout.log, stderr.log, submission.csv
  best/                # symlink-or-copy of the validation-best node
  final/submission.csv # what we actually submit
  report/index.html    # D's dashboard
  RESULTS.md           # D's results + resource table
```

---

## 6. Run modes (Role A's CLI, everyone uses them)

| Mode | Iterations | Data | LLM | Purpose |
|---|---|---|---|---|
| `smoke` | 3 | `--subsample 0.02` | stubbed, canned responses | CI, runs in < 60 s, must always pass |
| `dev` | 8 | `--subsample 0.2` | real, cheap model | daily integration check |
| `official` | 50 | full | real | the scored run. Never babysit it. |

`make check` runs lint + unit tests + `smoke`. Green before every merge, no exceptions.

---

## 7. Environment

- Python 3.11, `uv` or `venv` + `requirements.txt`, pinned versions
- `ANTHROPIC_API_KEY` from env only. **Never commit a key, never log one, never print one.**
- All ML deps go in `requirements-pipeline.txt` — the sandbox installs from that, and the
  generated `pipeline.py` may only import from it. Adding a library there is a Role C call.
