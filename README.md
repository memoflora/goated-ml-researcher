# goated-ml-researcher

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**

An LLM-driven orchestrator that runs the ML engineering loop by itself: read the problem,
inspect the data, engineer features, train and tune, evaluate, reflect, revise — repeating up
to 50 iterations or 6 hours with as close to zero human intervention as we can manage. It
writes its own pipeline code every round.

AIDE-shaped ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)): ML engineering as code
optimisation, searched over a tree of solution programs.

**Give it a dataset and a problem statement and it does the rest.** The benchmark task is
KuaiRand-Pure, but the agent is not built around it — a new problem is a YAML file in
[`tasks/`](tasks/), not a patch. See [Running it on your own data](#running-it-on-your-own-data).

## The number to beat

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline — validation | 0.6674 | 0.5357 | **0.6016** |
| Official baseline — hidden test | 0.6610 | 0.5282 | **0.5946** |
| Our reproduction — validation ✅ | 0.6671 | 0.5358 | **0.6015** |
| Perfect ranking (attainable ceiling) | 1.0000 | 0.7289 | **0.8645** |

The ceiling is 0.8645, not 1.0 — 27.1% of hidden-test users have no positive label, so their
nDCG is 0 for any model. Random scoring sits at 0.4753.

## Running it on your own data

```bash
python -m orchestrator.run --list-tasks              # what is defined
python -m orchestrator.run --task kuairand-pure --mode dev
python -m orchestrator.run --task demo-regression --mode dev    # a different problem entirely
```

To point it at something new, write one file. This is the whole contract:

```yaml
# tasks/my-problem.yaml
name: my-problem
kind: regression            # regression | binary | multiclass | ranking
description: |
  What is being predicted, and from what. The agent reads this before it writes any
  code, so say what a row is and what the target means.

data:
  dir: data/my-problem
  file: everything.csv      # or files: {train: train.csv, test: test.csv}
  target: the_column_to_predict
  id_columns: [row_key]
  split: {strategy: random, valid_frac: 0.2, test_frac: 0.2, seed: 0}

metrics:
  primary: [rmse]           # what the search maximises (lower-is-better is negated)
  report:  [rmse, mae, r2]
```

Then `python -m orchestrator.run --task my-problem --mode dev`. Everything else is derived:

| Derived | How |
|---|---|
| The data card the agent reads | `orchestrator/profile.py` profiles the table — types, cardinalities, missingness, relationship to target, and warnings for constant, identifier-like and leakage-grade columns |
| The splits | materialised to CSV once, so the pipeline never re-derives them. `test.csv` is written **without the target column**, so no-peeking is enforced by absence, not by instruction |
| Scoring | `orchestrator/metrics.py` — RMSE, MAE, R², RMSLE, AUC, log loss, accuracy, F1, average precision, macro-F1, GAUC, nDCG@K, MAP@K, recall@K |
| The submission schema | from the task file; the sandbox and the validator both read it |
| The idea bank | `tasks/ideas/generic-tabular.yaml` unless the task names its own |

Split strategies: `predefined` (a file per split), `date` (filter a date column by range),
`random`, and `group` (holds out whole groups, never rows, so grouped metrics stay valid).

**Metric direction is handled for you.** The search always maximises, so a lower-is-better
metric is negated inside `primary` — you write `primary: [rmse]` and the agent reports
`primary: -20.03`. Nothing downstream has to know which way a metric runs.

## The team

Two ML engineers build the agent; two ML researchers decide what it knows and what it tries.
Each person owns exactly one judged criterion.

| | Role | Owns | Owns the score for |
|---|---|---|---|
| **A** | ML Engineer — Orchestrator & Run | `contracts.py` `core.py` `policy.py` `journal.py` `run.py` | Autonomy |
| **B** | ML Engineer — Agent Runtime & Sandbox | `agent.py` `sandbox.py` `Makefile` `.github/` | Robustness + Feasibility |
| **C** | ML Researcher — Data, Metrics & Evaluation | `evaluate.py` `datacard.py` `report.py` `data/` | Primary metric |
| **D** | ML Researcher — Method, Knowledge & Story | `ideas.yaml` `knowledge.py` `prompts/` `reference/` `docs/` | Innovation + Presentation |

Ownership is per file, not per directory. B owns the prompt plumbing; D owns the prompt text.

Open Claude Code in this repo and start every session with your letter:

```bash
/techjam-track2 A
```

That loads the problem digest, the frozen contracts, your section of the role file, and the
current checkpoint. The skill lives in [`.claude/skills/techjam-track2/`](.claude/skills/techjam-track2)
and is picked up automatically on clone.

## Documents

| | |
|---|---|
| [PLAN.md](PLAN.md) | Roles, the 72-hour schedule, standing rules, setup checklist |
| [STATUS.md](STATUS.md) | Shared scratchpad — update before every sync |
| [problem.md](.claude/skills/techjam-track2/references/problem.md) | Problem statement digest, scoring rubric, constraints |
| [starter-kit-findings.md](.claude/skills/techjam-track2/references/starter-kit-findings.md) | Verified baseline reproduction, measured dead ends, where the headroom is |
| [contracts.md](.claude/skills/techjam-track2/references/contracts.md) | Architecture, module layout, frozen interfaces, journal schema |
| [roles.md](.claude/skills/techjam-track2/references/roles.md) | All four roles: build order, acceptance tests, traps |

## Planned layout

One file per concern — no nested packages.

```
orchestrator/
  contracts.py    A   frozen dataclasses
  core.py         A   loop, tree, convergence, budget, resume
  policy.py       A   draft / improve / debug / explore selection
  journal.py      A   append-only JSONL + token & wall-clock accounting
  run.py          A   CLI
  agent.py        B   LLM client, prompt assembly, proposal parsing, repair
  sandbox.py      B   subprocess runner, timeouts, error classification
  evaluate.py     C   score() and validate()
  datacard.py     C   the EDA summary the LLM reads
  report.py       C   journal -> RESULTS.md + trajectory PNG
  knowledge.py    D   retrieve()
  ideas.yaml      D   the KuaiRand idea bank
  prompts/        D   system.md, draft.md, improve.md, repair.md (task-templated)
  taskspec.py     C   tasks/*.yaml -> TaskConfig
  metrics.py      C   the metric registry, with directions
  datasource.py   C   generic loading, splitting, materialising
  profile.py      C   automatic EDA for a dataset nobody has described
tasks/            C   one YAML per problem, plus tasks/ideas/ for per-task banks
tools/            -   make_demo_data.py, for the offline demo task
data/             C   splits and caches (gitignored)
reference/        D   hand-written calibration pipeline
runs/             -   per-run workspaces + journal.jsonl (gitignored)
docs/             D   Devpost writeup, diagram
tests/            all B owns the smoke run
```

## Setup

- [ ] Registered on **both** the Registration Form and Devpost — both are required
- [ ] `kuairand-starter-kit.zip` from the Lark information document, §2.4
- [ ] KuaiRand-Pure data from [kuairand.com](https://kuairand.com)
- [ ] `ANTHROPIC_API_KEY` in your environment — never committed
- [ ] Python 3.11 + venv

## The two rules that can lose us everything

1. **No external training data.** KuaiRand only — no augmenting, joining, or pre-training on any
   other dataset, and no pretrained weights trained on these benchmarks' test labels. The single
   disqualifying rule.
2. **Nothing after H+60 leaves us without a submittable project.**
