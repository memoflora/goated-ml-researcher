# goated-ml-researcher

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**

This is an autonomous ML research agent: an LLM writes a complete `pipeline.py`, a sandbox
runs it, an evaluator scores its submission, a search policy decides what to try next, and
an append-only journal records every hypothesis, failure and score. It is AIDE-shaped
([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)) — ML engineering treated as code
optimisation over a tree of solution programs — and it runs unattended for up to 50
iterations or 6 hours with no human in the loop. The benchmark is KuaiRand-Pure, but the
dataset is configuration rather than code: a new problem is a YAML file in [`tasks/`](tasks/),
not a patch.

---

## Where we stand against the benchmark

The bar is the organisers' Factorization Machine baseline. Everything below is
**validation** — we have never scored the hidden test set, and the guard that stops us is
in the code, not in a promise.

| | primary | vs baseline | what it is |
|---|---|---|---|
| Random | 0.4834 | −0.118 | the floor |
| Item popularity | 0.5807 | −0.021 | a 20-line heuristic |
| **Official FM baseline** | **0.6016** | — | **the bar we must clear** |
| Our reproduction of it | 0.6015 | −0.0001 | proves our stack scores correctly |
| BPR-FM control (hand-written) | 0.6032 | +0.0016 | one human change, for calibration |
| **Best autonomous agent run** | **0.6189** | **+0.0173** | run `r20260831-0532`, 12 iterations |
| Oracle ceiling | 0.8484 | +0.247 | a *perfect* ranking, not 1.0 |

**The agent has beaten the baseline.** Its best run reached 0.6189 — **+0.0173**, about ten
times the gain of our hand-written control, and roughly 7% of the total headroom above the
baseline. It converged in 12 iterations for 124k tokens and 8.5 minutes, and it got there
with a LambdaRank pairwise model it chose and wrote itself.

### What is not yet true

**That run produced no submittable artifact.** It scored on `--split val` and then failed at
finalisation, because its `--split test` branch — a code path no development iteration ever
executes — used a pandas method removed in 2.x. The model was fine; the branch nothing had
ever run was not.

**Our only complete run so far is a bad one.** The first 50-iteration official run produced a
valid 170,588-row submission scoring **0.4839** — below random, and worse than item
popularity. The cause was ours, not the model's: a test-path check that disqualified any node
whose test branch failed. Sophisticated pipelines break there far more often than trivial
ones, so it systematically selected for triviality — it rejected a trained pairwise-nDCG model
at 0.4959 and crowned one that ranked by video ID with 2.2 seconds of "training". That check
no longer vetoes anything; it records the fault and repairs it at finalisation, against the
node that actually won.

So the honest position: **a model that clears the bar, and a submission pipeline that has not
yet carried one across.** The next official run is the one that tests whether those two facts
can hold at the same time.

### Caveats we are not hiding

- 0.6189 is **one run**. Draft temperature is 1.0, and a second dev run at the same settings
  scored nothing at all. We have not yet established variance.
- It trained on a **20% user subsample**. Scoring was on the full validation split, so the
  comparison is fair, but a full-data run may land elsewhere.
- **Validation is not hidden test.** The baseline itself drops 0.0070 between the two.

---

## Quickstart — no API key, no dataset, under five minutes

Everything below runs offline. The `smoke` mode pins the LLM, the sandbox and the evaluator
to stubs, which is exactly how CI runs it, so you can see the whole loop turn without a key
and without downloading 195 MB of KuaiRand.

```bash
git clone <this repo> && cd goated-ml-researcher
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m orchestrator.run --list-tasks          # what problems are defined
python -m orchestrator.run --task kuairand-pure --mode smoke
```

That prints a JSON summary and the path to the run's journal:

```
{
 "run_id": "r20260831-0341",
 "stop_reason": "max_iters",
 "iterations": 3,
 "best_node": "n002",
 ...
}

journal: runs/r20260831-0341/journal.jsonl
final:   runs/r20260831-0341/final/submission.csv
```

Then turn that journal into the graded deliverable:

```bash
python -m orchestrator.report runs/<run_id>      # writes RESULTS.md + trajectory.png
```

> The scores a `smoke` run reports come from the stub evaluator. They prove the plumbing
> works end to end; they say nothing about the dataset. The measured numbers are in
> [The numbers](#the-numbers) below.

### A second, completely different problem — still no key

The orchestrator is not KuaiRand-shaped. This generates a synthetic tabular regression
problem (continuous target, no groups, RMSE, a different submission schema) and runs the same
loop over it:

```bash
python tools/make_demo_data.py                   # writes data/demo-regression/listings.csv
python -m orchestrator.run --task demo-regression --mode smoke
```

## Running it for real

You need one LLM key and, for KuaiRand, the dataset.

```bash
# 1. a key — either provider works; agent.py adapts OpenAI to the Anthropic Messages shape
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY=...
# optional: TECHJAM_LLM={anthropic,openai,stub} to force a provider
# optional: TECHJAM_MODEL=<model id>  (required when using OpenAI — no model is guessed)
python -m orchestrator.models        # list the model ids your key can actually reach

# 2. the data (195 MB, gitignored — data/ is not in the repo)
mkdir -p data && cd data
curl -sL -o KuaiRand-Pure.tar.gz https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz         # -> data/KuaiRand-Pure/data
cd ..

# 3. run
python -m orchestrator.run --task kuairand-pure --mode dev        # 8 iterations, 20% subsample
python -m orchestrator.run --task kuairand-pure --mode official   # the scored run: 50 iters / 6 h
```

A `.env` file at the repo root works instead of exported variables; it is gitignored and the
journal redacts anything key-shaped that reaches it.

| Mode | Iterations | Data | LLM | Purpose |
|---|---|---|---|---|
| `smoke` | 3 | 2% subsample | stubbed | CI and this quickstart. No key, no dataset. |
| `dev` | 8 | 20% subsample | real | daily integration check |
| `official` | 50 | full | real | the scored run — 6 h ceiling, never babysat |

Useful flags — all of these are real, check them with `python -m orchestrator.run --help`:

```
--task NAME|PATH     a file in tasks/, or a path to one
--mode {smoke,dev,official}
--max-iters N        override the mode's iteration cap
--wall-clock 6h      6h | 90m | 3600s
--resume RUN_ID      restart from runs/<RUN_ID>/state.json
--seed N             --data-dir DIR --runs-dir DIR --subsample F
--timeout N          per-pipeline seconds     --token-budget N
--agent {auto,stub}  --sandbox {auto,stub}  --evaluator {auto,stub}
```

The three seam flags are how you run half the system offline: `--agent stub` keeps the real
sandbox and the real evaluator but spends no tokens.

## The numbers

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline — validation | 0.6674 | 0.5357 | **0.6016** |
| Official baseline — hidden test | 0.6610 | 0.5282 | **0.5946** |
| Our reproduction — validation | 0.6671 | 0.5358 | **0.6015** |
| BPR-FM reference (calibration control) | 0.6698 | 0.5365 | **0.6032** |
| Oracle ceiling — validation | 1.0000 | 0.6968 | **0.8484** |
| Oracle ceiling — hidden test | 1.0000 | 0.7289 | **0.8645** |
| Item popularity — hidden test | 0.6308 | 0.5121 | **0.5715** |
| Random — hidden test | 0.4996 | 0.4511 | **0.4753** |

`primary` is the equal-weighted mean of GAUC and nDCG@5.

**Read progress against 0.8645, not 1.0.** A *perfect* ranking of the hidden test scores
0.8645, because 27.1% of test users have no positive label and their nDCG is 0 for any model.
Against the validation oracle of 0.8484 the headroom above the baseline is 0.2468 — that is
the number an improvement should be measured against.

Five things we measured before the agent ever ran, all of which are encoded in
[`orchestrator/ideas.yaml`](orchestrator/ideas.yaml) and injected into every prompt:

1. **The test labels are physically on our disk.** KuaiRand-Pure is a public dataset;
   "hidden test" means the organisers score their own copy. So the no-peeking rule is ours to
   enforce, and we enforce it mechanically: `evaluate.score(..., "test")` refuses to run
   unless `ALLOW_TEST_SCORING=1` is set explicitly.
2. **More static features is a dead end.** All 13 feature fields scored 0.5940 against the
   5-field baseline's 0.5950 — inside noise, slightly worse.
3. **More capacity is a dead end.** Embedding k = 8 / 16 / 32 gives 0.5895 / 0.5902 / 0.5887.
   Flat.
4. **Ranking is within-user**, so a user-side first-order term contributes exactly zero to the
   ordering. User-side signal can only pay through a cross with the item side.
5. **Ours, and the one we would lead with:** with a pairwise ranking loss, *the pair-sampling
   weight matters more than the loss function, and it decides the sign of the result.* Same
   model, same loss, only the sampling changed: uniform users → 0.5982 (worse than baseline),
   users weighted by positive count → 0.6032 (better). GAUC averages per-user AUC weighted by
   positive count, so uniform sampling optimises a different quantity from the one being
   scored. An agent told only "use a ranking loss" implements the uniform version, measures a
   loss, and abandons the single most promising direction on the strength of a detail nobody
   thought to write down. `ideas.yaml` now writes it down.

The controlled experiment behind #5 is [`reference/bpr_fm.py`](reference/); it is never shown
to the agent and never submitted.

## What is in the box

```
orchestrator/
  run.py          the CLI, and the seam resolver that swaps stubs in for real modules
  core.py         the loop: tree, convergence, budget, checkpoint/resume, finalisation
  policy.py       what to try next — debug-first, draft phase, explore, greedy improve
  contracts.py    the frozen dataclasses every module shares
  journal.py      append-only JSONL writer + token and wall-clock accounting
  agent.py        LLM client, prompt assembly, proposal parsing, the repair loop
  sandbox.py      runs pipeline.py in a subprocess; timeouts, kills, error classification
  evaluate.py     score() and validate() — delegates every metric to the starter kit
  metrics.py      the metric registry; every metric declares its direction
  splits.py       cached access to the official KuaiRand splits, in authoritative row order
  taskspec.py     tasks/*.yaml -> TaskConfig; the file that makes a dataset configuration
  datasource.py   generic loading, splitting and split materialisation for any table
  profile.py      automatic EDA, so the agent can face a dataset nobody has described
  datacard.py     the markdown EDA summary the LLM actually reads (facts only, no advice)
  knowledge.py    retrieve() — rule-based selection over the idea bank
  ideas.yaml      the KuaiRand idea bank: 33 ideas in 5 tiers (17 cited), plus 4 dead ends
  prompts/        system.md, draft.md, improve.md, repair.md — templated per task
  report.py       journal.jsonl -> RESULTS.md + trajectory.png
tasks/            one YAML per problem; tasks/ideas/ holds per-task idea banks
tools/            make_demo_data.py — the offline demo fixture
reference/        hand-written calibration pipelines. Never submitted, never shown to the agent
vendor/           the organisers' starter kit — the sole authority on the metrics
tests/            281 tests; tests/stubs/ is what smoke mode runs on
```

### Where the output lands

```
runs/<run_id>/
  config.json          resolved task + CLI args + git sha + model id
  journal.jsonl        the graded log — one JSON object per line, flushed on every write
  state.json           tree + budget snapshot, rewritten atomically each iteration
  summary.json         the same JSON the CLI prints when the run ends
  interventions.md     every human touch, timestamped. The target is an empty file
  nodes/n000/          pipeline.py, stdout.log, submission.csv — one directory per attempt
  best/                a copy of the validation-best node
  final/submission.csv what we would submit
  RESULTS.md           written by `python -m orchestrator.report`
  trajectory.png       the headline chart
```

`RESULTS.md` regenerates from `journal.jsonl` alone, at any moment — mid-run, or after a
crash — so a run that dies at hour four still has a current deliverable.

## Tests

```bash
python -m pytest tests/ -q
```

281 tests. How many *skip* depends on your box, and every skip states its reason — run with
`-rs` to read them. Expect skips for: the KuaiRand-dependent tests when the dataset is not
downloaded, the POSIX process-group and SIGKILL tests on Windows, and the provider tests when
`openai` is not installed.

Install `requirements.txt` before running them. The provider-selection tests import
`anthropic` and `openai` directly, and they *fail* rather than skip if one of the two is
present and the other is missing.

```bash
python -m pytest tests/test_knowledge.py -q     # the idea bank and the prompt text
python -m pytest tests/test_faults.py -q        # fault injection: every error class, end to end
python -m ruff check orchestrator tests         # lint
```

On POSIX there is a Makefile: `make check` is lint + tests + a 3-iteration stubbed smoke run,
and it is the merge gate. `make help` lists the rest.

## Pointing it at your own data

To run the agent on a new problem, write one file. This is the whole contract:

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
| The data card the agent reads | `profile.py` profiles the table — types, cardinalities, missingness, relationship to target, and warnings for constant, identifier-like and leakage-grade columns |
| The splits | materialised to CSV once, so the pipeline never re-derives them. `test.csv` is written **without** the target column, so no-peeking is enforced by absence rather than by instruction |
| Scoring | `metrics.py` — RMSE, MAE, R², RMSLE, AUC, log loss, accuracy, F1, average precision, macro-F1, GAUC, nDCG@K, MAP@K, recall@K |
| The submission schema | from the task file; the sandbox, the prompts and the validator all read it |
| The idea bank | `tasks/ideas/generic-tabular.yaml` unless the task names its own |

Split strategies: `predefined` (a file per split), `date` (filter a date column by range),
`random`, and `group` (holds out whole groups, never rows, so grouped metrics stay valid).

**Metric direction is handled for you.** The search always maximises, so a lower-is-better
metric is negated inside `primary` — you write `primary: [rmse]` and the agent reports
`primary: -20.03`. Nothing downstream has to know which way a metric runs.

A task file *describes, it never instructs*. No hyperparameters, no "try gradient boosting" —
what to try is the agent's job, informed by the idea bank.

## Documents

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Data flow, the module boundaries, and why each one is where it is |
| [docs/devpost.md](docs/devpost.md) | The submission writeup |
| [PLAN.md](PLAN.md) | Roles, the 72-hour schedule, standing rules |
| [STATUS.md](STATUS.md) | Shared scratchpad — updated before every sync |
| [problem.md](.claude/skills/techjam-track2/references/problem.md) | Problem statement digest, scoring rubric, constraints |
| [starter-kit-findings.md](.claude/skills/techjam-track2/references/starter-kit-findings.md) | Verified baseline reproduction, measured dead ends, where the headroom is |
| [contracts.md](.claude/skills/techjam-track2/references/contracts.md) | Frozen interfaces, journal schema, risk register |
| [reference/README.md](reference/README.md) | The BPR-FM calibration experiment and what it found |

## The team

Two ML engineers build the agent; two ML researchers decide what it knows and what it tries.
Each person owns exactly one judged criterion.

| | Role | Owns | Owns the score for |
|---|---|---|---|
| **A** | ML Engineer — Orchestrator & Run | `contracts.py` `core.py` `policy.py` `journal.py` `run.py` | Autonomy |
| **B** | ML Engineer — Agent Runtime & Sandbox | `agent.py` `sandbox.py` `Makefile` `.github/` | Robustness + Feasibility |
| **C** | ML Researcher — Data, Metrics & Evaluation | `evaluate.py` `datacard.py` `report.py` `taskspec.py` `metrics.py` `datasource.py` `profile.py` `splits.py` `tasks/` | Primary metric |
| **D** | ML Researcher — Method, Knowledge & Story | `ideas.yaml` `knowledge.py` `prompts/` `reference/` `docs/` | Innovation + Presentation |

Ownership is per file, not per directory. B owns the prompt plumbing; D owns the prompt text.

Open Claude Code in this repo and start every session with your letter:

```bash
/techjam-track2 A
```

That loads the problem digest, the frozen contracts, your section of the role file, and the
current checkpoint. The skill lives in [`.claude/skills/techjam-track2/`](.claude/skills/techjam-track2)
and is picked up automatically on clone.

## The two rules that can lose us everything

1. **No external training data.** KuaiRand only — no augmenting, joining, or pre-training on
   any other dataset, and no pretrained weights trained on these benchmarks' test labels. It
   is the single disqualifying rule, it is stated in the agent's system prompt, and no
   generated pipeline is allowed network access.
2. **No hidden-test access during development.** Train and validation only. Scored iterations
   run `--split test` exactly never; the final submission is the validation-best node rerun
   once with `--split test`, and it is validated without ever reading a test label.
