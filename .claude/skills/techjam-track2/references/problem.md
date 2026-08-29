# Track 2 — Autonomous ML Research Agent for Recommender Systems

Source: TikTok TechJam 2026 Information Document (problem statement last updated 27 Aug 2026).
This is a digest. Where it disagrees with the official doc, the official doc wins.

## The task

Design and implement an autonomous ML research agent. For the benchmark it must, on its own:

1. **Reproduce the official baseline.** Stand up a working end-to-end pipeline that reaches
   the organiser-provided reference score. A starter pipeline the agent builds for itself is
   an internal step, not the reference we are scored against.
2. **Iterate on the pipeline.** Draw on established methods from industry and academia and
   apply them in code, across *any* stage — data, features, model, training, evaluation.
   Development uses train + validation only. It never sees the hidden test set.
3. **Improve over the baseline.** Drive validation score above the official baseline over
   repeated iterations. Improvement need not be monotonic, but the trend must be clearly
   sustained. Final ranking is computed once, on the hidden test set, from the submission
   the agent designates as final.

Writing the code for each stage is the agent's job, not something we hand it.

## The MLE loop the agent must automate

```
read problem -> inspect data (EDA) -> engineer features -> train + tune -> evaluate
       ^                                                                     |
       +------------------------ reflect + revise <--------------------------+
```

## Benchmarks

| Dataset | Status | Weight |
|---|---|---|
| **KuaiRand-Pure** | **Required** | 100% of the primary metric |
| KuaiRand-1k | Bonus | extra credit only |
| KuaiRand-27k | Bonus | extra credit only |

Skipping the bonus datasets does not reduce the KuaiRand-Pure score. Data: https://kuairand.com

## Starter kit

Download `kuairand-starter-kit.zip` from the Lark information document (Section 2.4).

- numpy only — no torch / pandas / scikit-learn in the kit itself
- `python3 baseline.py --model fm` reproduces the official baseline in ~40 s on one CPU core
- `evaluate.py` — the exact scoring code; model-agnostic, takes `(user_ids, labels, scores)`
- `submit.py --make` generates a runnable example; `submit.py --check` validates a submission

### Fixed splits (date-based, from the two standard pure logs)

| Split | Dates | Rows |
|---|---|---|
| train | 20220408–20220421 | 1,141,112 |
| validation | 20220422–20220428 | 124,909 |
| test (hidden) | 20220429–20220508 | 170,588 |

### The official baseline — this is the number to beat

Factorization Machine, k=16, lr=0.001, 5 categorical fields, numpy only, ~40 s CPU.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| **Hidden test (published)** | 0.6610 | 0.5282 | **0.5946** (mean of 5 seeds, std 0.0008) |
| **Validation** | 0.6674 | 0.5357 | **0.6016** |

Harness self-check rungs: random scoring → primary 0.4753; item popularity → primary 0.5715.

### Reading the numbers — do not chase 1.0

27.1% of hidden-test users have no positive label (nDCG = 0 for any model) and 9.2% are
all-positive. A **perfect** ranking reaches only GAUC 1.0000 / nDCG@5 0.7289 / **primary
0.8645**. Random is 0.4753. The baseline's 0.5946 already captures ~31% of the attainable
range. Judge progress against **0.8645**, not 1.0.

### Metric conventions (pinned — do not reimplement, call `evaluate.py`)

- users with zero positives count as nDCG = 0 and are **included** in the average
- GAUC counts only users with `0 < positives < impressions`, weighted by positive count
- nDCG gain = `2^rel − 1`
- primary = equal-weighted mean of GAUC and nDCG@5

### Submission format

CSV with header `row_id,user_id,video_id,score`, one line per evaluation-split row.

- `row_id` — 0-based, strictly increasing index into the split as produced by `data.load()`.
  Required because `(user_id, video_id)` is **not unique**: 3.06% of test rows are repeated
  pairs, up to 12 times.
- `user_id` / `video_id` — redundant, used only to verify alignment
- `score` — any real number, only relative order matters. NaN / Inf are rejected.

## Constraints and scope

**In scope:** any open-source library (PyTorch, RecBole, TorchRec, LightGBM, …), any papers,
public solutions or pretrained weights, changes to any pipeline stage — not just the model.

**Out of scope (hard rules):**

- **No external training data.** Only the KuaiRand datasets. No augmenting, joining, or
  pre-training on any other dataset. No pretrained weights trained on these benchmarks'
  test labels. This is the single rule that keeps ranking fair — violating it is fatal.
- No hidden-test access during development. Train + validation only.

**Limits:**

- **50 iterations** per benchmark run (hard cap)
- **6 h wall-clock** ceiling per run (backstop)
- Convergence rule: **eps = 0.002, N = 3** — converged when validation primary has not
  improved by more than eps over the last 3 consecutive iterations. (eps is about 2.5 sigma
  of the baseline's 0.0008 five-seed std.)
- The scored submission is the **validation-best checkpoint at the convergence point**,
  evaluated once on hidden test.
- Compute is deliberately *not* the binding constraint: 100 iterations of the official
  baseline is ~28 min on one CPU core, no GPU. GPU-hours and LLM tokens are reported for
  Feasibility scoring, not capped.

## Deliverables

1. Written project description (Devpost)
2. Public code / GitHub repository
3. **Run and iteration logs**, plus a short summary reporting the **number of manual
   interventions** during the run
4. Final submission and results summary:
   - final model output/checkpoint for KuaiRand-Pure in the starter-kit schema
   - results table: validation-best GAUC / nDCG@5 and the **absolute delta over the official
     baseline**; include bonus benchmarks if attempted
   - **reported resource usage** to reach convergence: total LLM tokens (input + output),
     total agent wall-clock, iterations used (out of 50). GPU-hours if any GPU was used.

## How we are scored — build to this

| Criterion | What it measures | Where it is won in our repo |
|---|---|---|
| **Technical Execution — Primary metric** | Equal-weighted mean of each metric's *absolute improvement* over the official baseline, on hidden test. KuaiRand-Pure = 100%. | **C** (correct, fast measurement) + **D** (the ideas) + **A** (search policy) |
| **Technical Execution — Robustness** | Not failure count — how the agent *handles* a failed step: recover, retry, route around. Long runs must not crash, stall or diverge. | **B** (repair loop) + **A** (dead-node routing) |
| **Innovation & Problem Insight** | What the agent chose to target across the *full* stack and **why**. Originality in drawing on published methods; beyond naive baseline tweaks. Judged on the choice, not the implementation. | **D** (cited idea bank + prompts demanding a `hypothesis`) |
| **Impact & Relevance — Autonomy** | How much of the loop the agent drives itself. Measured primarily by **number of manual interventions**. Fewer is higher; fully autonomous scores highest. | **A** (unattended runs, resume) — but every manual fix is everyone's bug |
| **Feasibility & Practicality — Resource** | Total input+output tokens and agent wall-clock to reach convergence. Graded in three coarse tiers, and **only scored among submissions that beat the baseline on hidden test**. | **B** (prompt and caching discipline) + **A** (accounting) |

Note the gate on the last row: **beat the baseline first, then be cheap.** An agent that
stops after three iterations to look cheap scores worst.

## Overall hackathon criteria (applied on top of the track criteria)

Technical Execution · Innovation & Problem Insight · Impact & Relevance · Feasibility &
Practicality · Presentation & Communication (pitch quality — final event only).

## Key dates

- **29 Aug 12:00 SGT → 1 Sep 12:00 SGT** — 72-hour challenge, submissions on Devpost
- **1 Sep 12:00 SGT** — hard submission deadline, no late entries
- 1 Sep 15:00 → 4 Sep 15:00 — People's Choice public voting on Devpost
- 8 Sep finalists · 11 Sep grand final at TikTok Singapore · 15 Sep winners

## Prior art the agent should know about

- MLE-bench (OpenAI, 2024), arXiv:2410.07095 — benchmark of 75 Kaggle competitions for ML agents
- AIDE (Weco AI, 2025), arXiv:2502.13138 — ML engineering as code optimisation via tree search.
  **Closest prior art to what we are building; read this one first.**
- AI-Scientist-v2 (Sakana, 2025), arXiv:2504.08066 — agentic tree search for research
- Counteracting Duration Bias in Video Recommendation (KDD 2024), https://github.com/hyz20/CWM —
  censored-regression loss on watch time. Optional advanced reference, **not** the baseline;
  ships no Recall implementation and needs torch==1.6.0.

## Domain primer (from Appendix A of the doc)

- Industrial recommenders are a funnel: **recall → pre-ranking → ranking → re-ranking**.
  This challenge lives in **ranking**.
- Core task framing is CTR-style: P(feedback | impression).
- **KuaiRand provides 12 feedback signals** even though only `long_view` is scored — a
  multi-task model can learn from several jointly. Balance shared vs task-specific
  parameters (the "seesaw" problem).
- Feature basics: high-cardinality ID features → embeddings; feature crossing (user × category);
  FM and DeepFM automate crossing.
