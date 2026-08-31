# goated-ml-researcher

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**

An LLM writes a complete `pipeline.py`, a sandbox runs it, an evaluator scores the
submission, a search policy picks what to try next, and an append-only journal records every
hypothesis, failure and score. AIDE-shaped ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)):
ML engineering as code optimisation over a tree of solution programs, running unattended for
up to 50 iterations or 6 hours. The benchmark is KuaiRand-Pure, but a dataset is
configuration, not code — a new problem is a YAML file in [`tasks/`](tasks/).

---

## Where we stand

All numbers are **validation**. We have never scored the hidden test set, and the guard is in
the code (`ALLOW_TEST_SCORING`), not in a promise.

| | primary | vs baseline |
|---|---|---|
| Random | 0.4834 | −0.118 |
| Item popularity | 0.5807 | −0.021 |
| **Official FM baseline** | **0.6016** | — |
| Our reproduction of it | 0.6015 | −0.0001 |
| BPR-FM control (hand-written) | 0.6032 | +0.0016 |
| **Best autonomous run** — gpt-4o | **0.6189** | **+0.0173** |
| Oracle ceiling (a *perfect* ranking) | 0.8484 | +0.247 |

**The agent beat the baseline by +0.0173** — ~10× our hand-written control, ~7% of the
headroom above baseline — converging in 12 iterations for 124k tokens and 8.5 minutes. It
chose and wrote a LambdaRank pairwise model itself:

> *"The current model uses pointwise loss, which doesn't align well with ranking metrics like
> GAUC and nDCG@5. By switching to a pairwise loss function and weighting the pairs by the
> change in nDCG@5 they cause…"* — `n011`, run `r20260831-0532`

That is the organisers' own top-ranked untested direction, reached by the agent.

**Read progress against 0.8484, not 1.0.** 27.1% of test users have no positive label, so
their nDCG is 0 for any model; a perfect ranking scores 0.8645 on test, 0.8484 on validation.

### What is not yet true

- **That run produced no submittable artifact.** It failed at finalisation on its
  `--split test` branch — a path no development iteration executes. Fixed since.
- **Our only complete run scored 0.4839**, below random. The cause was ours: a test-path
  check that disqualified any node whose test branch failed, which selected for triviality.
  It now records the fault and repairs it at finalisation instead of vetoing.
- **0.6189 is one run** at draft temperature 1.0; a second dev run at identical settings
  scored nothing. Variance is unmeasured.
- Validation is not hidden test — the baseline itself drops 0.0070 between them.

A model that clears the bar, and a submission pipeline that has not yet carried one across.

### Five things measured before the agent ever ran

All encoded in [`ideas.yaml`](orchestrator/ideas.yaml) and injected into every prompt.

1. **The test labels are on our disk.** KuaiRand-Pure is public; "hidden test" means the
   organisers score their own copy. So we enforce no-peeking mechanically.
2. **More static features is a dead end** — all 13 fields scored 0.5940 vs the 5-field 0.5950.
3. **More capacity is a dead end** — k = 8/16/32 gives 0.5895/0.5902/0.5887.
4. **Ranking is within-user**, so a user-side first-order term contributes exactly zero.
   User-side signal only pays through a cross with the item side.
5. **Ours:** with a pairwise loss, *the pair-sampling weight matters more than the loss and
   decides the sign*. Uniform users → 0.5982 (worse than baseline); weighted by positive count
   → 0.6032 (better). GAUC averages per-user AUC weighted by positive count, so uniform
   sampling optimises a different quantity. An agent told only "use a ranking loss" builds the
   uniform version, measures a loss, and abandons the most promising direction. The controlled
   experiment is [`reference/bpr_fm.py`](reference/) — never shown to the agent, never submitted.

---

## Quickstart — no key, no dataset, five minutes

`smoke` pins the LLM, sandbox and evaluator to stubs. It is how CI runs.

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m orchestrator.run --list-tasks
python -m orchestrator.run --task kuairand-pure --mode smoke
python -m orchestrator.report runs/<run_id>      # -> RESULTS.md + trajectory.png
```

A `smoke` run's scores come from the stub evaluator: they prove the plumbing, not the model.

The orchestrator is not KuaiRand-shaped — this is a synthetic regression problem with a
different target, metric and submission schema, through the same loop:

```bash
python tools/make_demo_data.py
python -m orchestrator.run --task demo-regression --mode smoke
```

To exercise the whole loop against **real data** with no key and no tokens, `--agent replay`
serves canned real pipelines and reaches 0.5807:

```bash
python -m orchestrator.run --task kuairand-pure --mode smoke --agent replay \
       --sandbox auto --evaluator auto --max-iters 3 --subsample 1.0
```

## Running it for real

```bash
# 1. a key. OpenAI or Anthropic — agent.py adapts OpenAI to the Anthropic Messages shape.
export OPENAI_API_KEY=...            # or ANTHROPIC_API_KEY
export TECHJAM_MODEL=gpt-4o          # required for OpenAI; no model is ever guessed
python -m orchestrator.models        # what your key can actually reach

# 2. the data (195 MB, gitignored)
mkdir -p data && cd data
curl -sL -o KuaiRand-Pure.tar.gz https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz && cd ..

# 3. run
python -m orchestrator.run --task kuairand-pure --mode dev        # 8 iters, 20% subsample
python -m orchestrator.run --task kuairand-pure --mode official   # 50 iters / 6 h
```

A gitignored `.env` works instead of exports; the journal redacts anything key-shaped.
Set `TECHJAM_FALLBACK_LLM` / `TECHJAM_FALLBACK_MODEL` for a second provider — failover
happens on auth, model-not-found or 429/5xx after backoff, **never** on a bad proposal, and
`summary.json` records which provider actually served.

| Mode | Iterations | Data | LLM |
|---|---|---|---|
| `smoke` | 3 | 2% | stubbed — no key, no dataset |
| `dev` | 8 | 20% | real |
| `official` | 50 | full | real, 6 h ceiling, never babysat |

Flags: `--max-iters --wall-clock 6h --resume RUN_ID --seed --data-dir --runs-dir --subsample
--timeout --token-budget --agent {auto,stub,replay} --sandbox {auto,stub} --evaluator {auto,stub}`.
The seam flags run half the system offline. Full list: `--help`.

## What is in the box

```
orchestrator/
  run.py         the CLI and the seam resolver that swaps stubs for real modules
  core.py        the loop: tree, convergence, budget, checkpoint/resume, finalisation
  policy.py      what to try next — debug-first, draft phase, explore, greedy improve
  contracts.py   the frozen dataclasses every module shares
  journal.py     append-only JSONL + token and wall-clock accounting
  agent.py       LLM clients (Anthropic/OpenAI/Gemini), prompts, parsing, repair loop
  sandbox.py     runs pipeline.py in a subprocess; timeouts, kills, error classification
  evaluate.py    score() and validate() — every metric delegates to the starter kit
  metrics.py     metric registry; every metric declares its direction
  splits.py      cached official splits, in authoritative row order
  taskspec.py    tasks/*.yaml -> TaskConfig — what makes a dataset configuration
  datasource.py  generic loading, splitting, materialisation for any table
  profile.py     automatic EDA, for a dataset nobody has described
  datacard.py    the markdown summary the LLM reads (facts only, no advice)
  knowledge.py   retrieve() — rule-based selection over the idea bank
  ideas.yaml     33 ideas in 5 tiers (17 cited) + 4 measured dead ends
  prompts/       system, draft, improve, repair + skeleton.py, templated per task
  report.py      journal.jsonl -> RESULTS.md + trajectory.png
vendor/          the organisers' starter kit — sole authority on the metrics
reference/       hand-written controls. Never submitted, never shown to the agent
```

Each run writes `runs/<id>/` containing `journal.jsonl` (the graded log), `summary.json`,
`state.json` (atomic, resumable), `interventions.md` (target: empty), `nodes/nNNN/` per
attempt, `best/`, `final/submission.csv`, `RESULTS.md` and `trajectory.png`. `RESULTS.md`
regenerates from the journal alone at any moment, so a run that dies at hour four still has a
current deliverable.

## Tests

```bash
python -m pytest tests/ -q          # 359 tests
python -m ruff check orchestrator tests
```

Every skip states its reason (`-rs`): KuaiRand tests without the dataset, POSIX
process-group tests on Windows, provider tests without that SDK installed. Install
`requirements.txt` first — the provider tests *fail* rather than skip if one SDK is present
and the other is missing. On POSIX, `make check` is lint + tests + a stubbed smoke run.

## Your own data

One file. Everything else is derived.

```yaml
# tasks/my-problem.yaml
name: my-problem
kind: regression            # regression | binary | multiclass | ranking
description: |
  What is being predicted and from what. The agent reads this before writing code.
data:
  dir: data/my-problem
  file: everything.csv      # or files: {train: train.csv, test: test.csv}
  target: the_column_to_predict
  id_columns: [row_key]
  split: {strategy: random, valid_frac: 0.2, test_frac: 0.2, seed: 0}
metrics:
  primary: [rmse]           # the search maximises; lower-is-better is negated for you
  report:  [rmse, mae, r2]
```

`profile.py` writes the data card, splits are materialised once with `test.csv` written
**without** the target column (no-peeking by absence, not instruction), `metrics.py` covers
RMSE/MAE/R²/RMSLE/AUC/log-loss/accuracy/F1/AP/macro-F1/GAUC/nDCG@K/MAP@K/recall@K, and split
strategies are `predefined`, `date`, `random` and `group`. A task file **describes, never
instructs** — no hyperparameters, no "try gradient boosting". What to try is the agent's job.

## Documents

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Data flow and why each module boundary is where it is |
| [docs/devpost.md](docs/devpost.md) | The submission writeup |
| [STATUS.md](STATUS.md) | Run history and what each one taught |
| [PLAN.md](PLAN.md) | Roles and the 72-hour schedule |
| [reference/README.md](reference/README.md) | The BPR-FM calibration experiment |
| [`.claude/skills/techjam-track2/`](.claude/skills/techjam-track2) | Problem digest, frozen contracts, starter-kit findings, per-role briefs |

Team: two ML engineers own the machine (`core`/`policy`/`journal`/`run`, and
`agent`/`sandbox`/CI); two ML researchers own what it knows (`evaluate`/`metrics`/`datacard`,
and `ideas.yaml`/`prompts`/`docs`). Ownership is per file — B owns the prompt plumbing, D owns
the prompt text. Start a session with `/techjam-track2 <A|B|C|D>`.

## The two rules that can lose us everything

1. **No external training data.** KuaiRand only. It is the single disqualifying rule, it is in
   the agent's system prompt, and no generated pipeline gets network access.
2. **No hidden-test access during development.** Scored iterations run `--split test` never;
   the final submission is the validation-best node rerun once, validated without reading a
   single test label.
