# Architecture and frozen contracts

## What we are building

An **autonomous ML research agent**. An orchestrator drives a closed loop: an LLM writes a
complete `pipeline.py`, a sandbox runs it, an evaluator scores its submission, a search policy
decides what to try next, and a journal records the trajectory.

**The problem is configuration.** `tasks/<name>.yaml` names the data, target, split, metrics
and submission schema; the data card, evaluator, prompts, sandbox checks and report all read
their facts from there. KuaiRand-Pure is one such file and is 100% of the score — it keeps the
organisers' own loader and evaluation code, so nothing about how it is scored has changed.

The shape is AIDE-like (arXiv:2502.13138): **treat ML engineering as code optimisation and
search over a tree of solution programs.** Proven, easy to explain to judges, and it splits
cleanly across four people.

```
   tasks/*.yaml (C) ──► TaskConfig ──► everything below reads its facts from here
                        │
   ideas.yaml (D) ─┐
                   ├─► Context ─► agent.py (B) ─► Proposal ─► pipeline.py
   datacard.py (C) ─┘      ▲            │                          │
                           │            │ repair on error          ▼
                     core.py (A)        ▼                     sandbox.py (B)
                    loop + policy ◄── ExecResult ◄─────────────────┘
                           │
                           ├─► evaluate.py (C) ─► metrics
                           └─► journal.py (A) ─► journal.jsonl ─► report.py (C) ─► RESULTS.md
```

## Flat module layout

One file per concern. No nested packages — this is a 72-hour project.

```
orchestrator/
  contracts.py    A   the frozen dataclasses below
  core.py         A   loop, tree, convergence, budget, checkpoint/resume
  policy.py       A   draft / improve / debug / explore selection
  journal.py      A   append-only JSONL writer + token & wall-clock accounting
  run.py          A   the CLI
  agent.py        B   LLM client, prompt assembly, proposal parsing, repair loop
  sandbox.py      B   subprocess runner, timeouts, error classification
  evaluate.py     C   score() and validate(), per task
  datacard.py     C   the EDA summary the LLM reads (hand-tuned for KuaiRand, generated otherwise)
  report.py       C   journal -> RESULTS.md + trajectory PNG
  splits.py       C   the KuaiRand fast path, through the starter kit's loader
  taskspec.py     C   tasks/*.yaml -> TaskConfig
  metrics.py      C   metric registry; every metric declares a direction
  datasource.py   C   generic loading, splitting, split materialisation
  profile.py      C   automatic EDA, for a dataset nobody has hand-described
  knowledge.py    D   retrieve()
  ideas.yaml      D   the KuaiRand idea bank
  prompts/        D   system.md, draft.md, improve.md, repair.md — task-templated
tasks/            C   one YAML per problem; tasks/ideas/ holds per-task idea banks
tools/            -   make_demo_data.py (offline demo fixture)
data/             C   splits and caches (gitignored)
reference/        D   our hand-written calibration pipeline
runs/             -   per-run workspaces + journal.jsonl (gitignored)
docs/             D   Devpost writeup, diagram
tests/            all everyone tests their own files; B owns the smoke run
```

---

## 1. The pipeline contract — the most important interface

Every solution the agent produces is **one self-contained file, `pipeline.py`**, in that node's
workspace. No packages, no plugin registry, no imports from our repo.

```
python pipeline.py --data-dir DIR --out-dir DIR --split {val,test} --seed N [--subsample F]
```

It must:

1. read the fixed splits from `--data-dir`
2. train on **train only** when `--split val`; on **train + validation** when `--split test`
3. write `<out-dir>/submission.csv` with the task's header — `row_id,user_id,video_id,score`
   for KuaiRand, `submission.columns` in the task file otherwise. `row_id` is always first
   and is always the key.
4. print exactly one stdout line `RESULT_JSON {...}` with at least
   `{"n_rows": int, "train_seconds": float, "notes": str}`
5. exit 0 on success, non-zero on failure, never prompt for input
6. honour `--subsample F` by sampling **whole groups** when the task has a group column
   (users, for KuaiRand), and rows otherwise. Row sampling silently breaks a grouped metric.

For a non-KuaiRand task, `--data-dir` points at **materialised splits**: `train.csv`,
`valid.csv`, `test.csv`, written once so the pipeline never re-derives a split and drifts
from the rows it is scored on. `test.csv` has the target column removed.

Why one file: trivial to sandbox, diff, revert, and the LLM can hold all of it in context.
Do not "improve" this into a framework.

**Scoring never happens inside `pipeline.py`.** The orchestrator scores `submission.csv` with
C's evaluator. This pins the metric conventions and keeps the agent honest.

---

## 2. `orchestrator/contracts.py` (owner: A, frozen after H+2)

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
    name: str                       # "kuairand-pure"
    data_dir: Path
    metrics: tuple[str, ...]        # ("gauc", "ndcg@5")
    baseline_val: dict[str, float]  # {"gauc":0.6674,"ndcg@5":0.5357,"primary":0.6016}
    baseline_test: dict[str, float] # {"gauc":0.6610,"ndcg@5":0.5282,"primary":0.5946}
    ceiling: float | None = 0.8645
    max_iters: int = 50
    wall_clock_s: int = 6 * 3600
    conv_eps: float = 0.002
    conv_n: int = 3

    # Added when the orchestrator stopped assuming KuaiRand. Every field has a
    # KuaiRand-shaped default, so a TaskSpec built the old way is unchanged.
    kind: str = "ranking"           # ranking | binary | multiclass | regression
    description: str = ""           # the problem statement; goes into every prompt
    primary_parts: tuple[str, ...] = ("gauc", "ndcg@5")
    submission_columns: tuple[str, ...] = ("row_id", "user_id", "video_id", "score")
    prediction_column: str = "score"
    seed_std: float | None = 0.0008
    config: object | None = None    # the parsed tasks/<name>.yaml


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
```

---

## 3. The four function-level seams

```python
# agent.py (B) — prompt text comes from prompts/*.md, owned by D
def draft(ctx: Context) -> Proposal: ...
def improve(ctx: Context, parent: Node) -> Proposal: ...
def repair(ctx: Context, node: Node) -> Proposal: ...

# sandbox.py (B)
def run(node: Node, *, split: str, seed: int, timeout_s: int,
        subsample: float | None = None) -> ExecResult: ...

# evaluate.py (C)  -- `task` is optional and defaults to KuaiRand, so the frozen
#                     two-argument shape still works everywhere.
def score(submission: Path, split: str, task=None) -> dict[str, float]: ...
def validate(submission: Path, split: str, task=None) -> tuple[bool, str]: ...

# datacard.py (C)
def data_card(task=None) -> str: ...  # markdown EDA summary, <= 3000 tokens

# taskspec.py (C)
def load_task(ref: str | Path) -> TaskConfig: ...   # "kuairand-pure" or a path

# metrics.py (C)
def compute(name, y_true, y_pred, groups=None) -> float: ...
def primary_of(metrics: dict, parts: tuple[str, ...]) -> float: ...  # always maximised

# knowledge.py (D)
def retrieve(*, tried: list[str], best_metrics: dict,
             budget_left: int, k: int = 5) -> list[Idea]: ...

# journal.py (A)
def emit(event: dict) -> None: ...   # appends one JSON line, flushes
```

`Context` is assembled by A and passed to B. A plain dataclass holding: `TaskSpec`, C's data
card string, the parent's code and metrics, D's top-K `Idea`s, a compact history (hypothesis +
metric delta only, never past code), and the remaining iteration/time/token budget.

---

## 4. The loop (A implements)

```
for iteration in 1..50, while wall_clock < 6h and not converged:
    kind, parent = policy.next_action(tree)
    proposal     = agent.draft|improve|repair(ctx, parent)
    node         = tree.add(parent, kind, proposal)
    write proposal.code -> node.workspace/pipeline.py
    result       = sandbox.run(node, split="val", seed=0, timeout_s=...)

    if not result.ok:
        journal(error)
        if node.repair_attempts < 3: schedule a debug node next iteration
        else: node.status = "dead"; journal(recovery, "route_around")
        continue

    ok, msg = evaluate.validate(result.artifacts["submission"], "val")
    if not ok: treat as error_class="contract" and repair

    node.metrics = evaluate.score(result.artifacts["submission"], "val")
    journal(eval); update best; check convergence
```

Three things that matter:

- **A failed node never stops the run.** Three repairs, then dead, then route around. That
  behaviour *is* the Robustness score.
- **Convergence is on validation primary**, over *scored* iterations only. Errors do not count
  toward N.
- **The final submission** is the validation-best node rerun with `--split test`, validated with
  `submit.py --check`. Once, at the end.

---

## 5. Journal schema — `runs/<run_id>/journal.jsonl`

One JSON object per line, appended and flushed immediately. **This file is a graded
deliverable** — judges read it to score Autonomy, Robustness and Innovation. Public API.

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
  "plan": ["add smoothed user CTR prior"],
  "idea_ids": ["T1.user-ctr-prior"],
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

`event` ∈ `run_start`, `data_card`, `proposal`, `exec`, `eval`, `error`, `recovery`, `prune`,
`best_updated`, `intervention`, `converged`, `run_end`. Irrelevant fields are `null` or omitted.
Never write anything that is not valid single-line JSON. Never log a secret.

---

## 6. Run directory

```
runs/<run_id>/
  config.json          resolved TaskSpec + CLI args + git sha + model id
  journal.jsonl        the graded log
  state.json           tree + budget snapshot, rewritten atomically each iteration
  interventions.md     every human touch, timestamped. Scored. Keep it empty.
  nodes/n000/          pipeline.py, stdout.log, stderr.log, submission.csv
  best/                copy of the validation-best node
  final/submission.csv what we submit
  RESULTS.md           C's generated results + resource table
  trajectory.png       C's headline chart
```

---

## 7. Run modes

| Mode | Iterations | Data | LLM | Purpose |
|---|---|---|---|---|
| `smoke` | 3 | `--subsample 0.02` | stubbed | CI, under 60 s, must always pass |
| `dev` | 8 | `--subsample 0.2` | real, cheap | daily integration check |
| `official` | 50 | full | real | the scored run. Never babysit it. |

`make check` = lint + unit tests + `smoke`. Green before every merge, no exceptions.

---

## 8. Environment

- Python 3.11, `venv` + pinned `requirements.txt`
- `ANTHROPIC_API_KEY` from env only. **Never commit, log, or print a key.**
- Generated `pipeline.py` may import only from `requirements-pipeline.txt`. Adding a library
  there is C's call.

---

## Changing a contract

Everything above is frozen after H+2. To change one: add a `## Contract change proposed` entry
in `STATUS.md`, keep the old shape working, and get one other person to ack before merging.

## Non-goals — say no to these

- A web UI or HTML dashboard. Nobody scores it; `RESULTS.md` plus one PNG is enough.
- Planner/coder/critic personas inside the agent. Burns scored tokens, adds failure modes.
  One well-prompted call per node.
- **Anything past the task layer.** `tasks/*.yaml` exists so a new dataset is a config file.
  A plugin system, a second orchestrator, or a UI on top of it is scope we do not have.
  KuaiRand-Pure is still 100% of the score.
- GPU or distributed training. The reference pipeline is ~28 min of one CPU core for 100
  iterations. Wall-clock is scored; a GPU will not help and may hurt.
- Bonus benchmarks before the required one converges and beats the baseline.

## Risk register

| Risk | Mitigation | Owner |
|---|---|---|
| Agent overfits validation, loses on hidden test | seed-averaged final scoring; prefer the simpler node within noise; never select on one seed | A + C |
| Run dies at hour 4 and nobody notices | atomic checkpoint + `--resume`; RESULTS.md regenerates from the journal at any time | A + C |
| LLM keeps proposing the same idea | `tried` ids passed to `retrieve()`; dedupe by normalised code hash; force explore after 3 flat iterations | A + D |
| Token cost blows the Feasibility tier | hard token budget in `Context`; prompt-cache the static block; never send data or full history | B |
| Generated code imports something uninstalled | `requirements-pipeline.txt` is the whitelist, stated in `system.md`; `import` errors trigger a fallback repair | B + D |
| We beat the baseline but the CSV is malformed | `submit.py --check` runs on every scored node, not just the final one | C |
| Nobody has time for the Devpost entry | D starts it at H+12 and keeps it current | D |
