# Role C — ML Core, Data, Evaluation and Knowledge

You own the ML substrate and, through the idea bank, what the agent actually thinks to try.
The **Primary metric** and **Innovation & Problem Insight** scores are mostly yours.

## You own

```
orchestrator/eval/evaluator.py    wraps the starter kit's evaluate.py
orchestrator/eval/validate.py     wraps submit.py --check
orchestrator/eval/data_card.py    the EDA summary the LLM reads
orchestrator/knowledge/ideas.yaml the idea bank
orchestrator/knowledge/kb.py      retrieve()
data/                             splits, caches (gitignored)
requirements-pipeline.txt         the whitelist of libraries generated code may import
reference/                        our own hand-written pipelines, for calibration
tests/test_evaluator.py, tests/test_kb.py
```

## You do not touch

`orchestrator/core/`, `orchestrator/search/` (A) · `orchestrator/agent/`,
`orchestrator/exec/` (B) · `orchestrator/report/`, `docs/` (D).

## First task (due H+6): reproduce the official baseline

Nothing else you do counts until this number is on the board.

1. Download `kuairand-starter-kit.zip` from the Lark doc (Section 2.4) and the KuaiRand-Pure
   data from https://kuairand.com
2. `python3 baseline.py --model fm` — should take about 40 s on one CPU core
3. Confirm **validation primary ≈ 0.6016** (GAUC 0.6674 / nDCG@5 0.5357)
4. Sanity-check the harness against the published rungs: random scoring → primary 0.4753,
   item popularity → primary 0.5715. If those two do not reproduce, the harness is wrong.
5. Post the numbers in `STATUS.md`. This is the reference the whole run is scored against.

**Never reimplement the metrics.** Call the starter kit's `evaluate.py`. The conventions are
pinned (zero-positive users count as nDCG 0 and are included; GAUC only over users with
`0 < positives < impressions`, weighted by positive count; gain = `2^rel − 1`). Getting these
subtly wrong means our whole validation signal is a lie.

## Second task (due H+12): the data card

A markdown summary, **under 3000 tokens**, that the LLM reads on every draft. It is the
agent's only view of the data, so it decides the quality of every hypothesis. Include:

- split sizes and date ranges; the label definition (`long_view`) and its base rate
- the field list with cardinalities, missing rates, and which are user / item / context
- **all 12 feedback signals** and their rates — the multi-task angle depends on the agent
  knowing these exist
- video duration distribution and the duration-bias note (long videos get fewer completions)
- user activity distribution: impressions per user, positives per user, the 27.1% zero-positive
  and 9.2% all-positive users, and what that means for the metric ceiling of 0.8645
- the exposure/repeat structure: 3.06% of test rows are repeated `(user_id, video_id)` pairs
- an explicit list of leakage traps: no future data, no test labels, no external data

Write it as facts, not advice. The ideas belong in the idea bank.

## Third task (rolling): the idea bank

`ideas.yaml`, each entry matching the `Idea` dataclass (`id`, `tier`, `title`, `summary`,
`citation`, `est_minutes`, `prerequisites`). Aim for **30–50 entries with real citations**.
This is what makes us a *research* agent rather than a hyperparameter tuner, and the citations
are exactly what the Innovation rubric rewards.

Suggested tiers:

- **T0 — baseline mastery.** FM hyperparameters (k, lr, epochs, L2), feature hashing, negative
  handling, epoch/early-stopping on validation, seed averaging.
- **T1 — feature engineering.** Smoothed user/item CTR priors (Bayesian smoothing), user and
  item impression counts, recency features, time-of-day and day-of-week, video duration
  buckets, user × category crosses, target encoding with out-of-fold computation.
- **T2 — stronger models.** LightGBM/XGBoost on engineered features; DeepFM; DCN-v2; xDeepFM;
  Wide & Deep. Blending a GBDT with an embedding model is historically the strongest
  cheap win on tabular CTR data.
- **T3 — recsys structure.** Multi-task learning over the 12 feedback signals (MMoE, PLE) with
  `long_view` as the scored head; user behaviour sequence features with DIN-style attention;
  item and user embedding pretraining on the training log only.
- **T4 — advanced and calibration.** Duration-bias correction via censored watch-time
  regression (CWM, KDD 2024, https://github.com/hyz20/CWM — note it needs torch==1.6.0 and
  ships no Recall implementation, so port the idea, not the repo); listwise/ranking losses
  aligned to nDCG; score calibration; ensembling across seeds and model families.

Each summary must be **actionable in one iteration** — 2 to 4 sentences an LLM can turn into
code without further research. "Use MMoE" is useless; "add a shared bottom of 2 dense layers
with one expert head per feedback signal, train all 12 heads with equal weight, score only
the `long_view` head" is usable.

`retrieve()` takes what has already been tried, the current best metrics and the remaining
budget, and returns top-K. Simple rules beat embeddings here: filter out tried ids, filter out
ideas whose prerequisites are unmet, prefer the lowest tier that still has untried entries,
and escalate a tier when the current one stops producing gains.

## Fourth task (due H+36): the reference pipeline

Hand-write a pipeline (T1-level features + LightGBM, say) that beats the baseline on
validation. **This is not for submission** — it is calibration. It tells us how much headroom
exists, so when the agent plateaus we know whether the bottleneck is the agent or the dataset.
Keep it in `reference/`, and keep it out of the agent's context.

## Fast-iteration harness

The agent gets 50 iterations and 6 hours. If each pipeline takes 20 minutes, we get 18. Make
training fast:

- cache parsed splits as `.npy` so every node does not re-parse CSVs
- a shared read-only feature cache keyed by feature-spec hash
- honour `--subsample F` by sampling **users**, not rows (row sampling breaks GAUC)
- keep `requirements-pipeline.txt` tight; every added library is install time on every node

## Acceptance tests — you are done when

- [ ] Official baseline reproduces within 0.002 of published validation primary
- [ ] Random and item-popularity rungs reproduce (0.4753 / 0.5715)
- [ ] `Evaluator.score()` is deterministic and agrees with `evaluate.py` exactly
- [ ] `validate()` rejects: wrong header, row-count mismatch, `row_id` gaps, misalignment,
      non-numeric / NaN / Inf scores
- [ ] Data card is under 3000 tokens and contains no advice, only facts
- [ ] Idea bank has 30+ entries, every T2+ entry carries a citation
- [ ] A full pipeline run on cached features completes in under 5 minutes

## Traps

- **No external training data. Ever.** Only KuaiRand. No pretrained weights touched by these
  benchmarks' test labels. This is the one rule that disqualifies.
- **Do not compute features on train + validation combined during development.** Validation
  must stay clean or our stopping signal is worthless.
- **Do not let target encoding leak.** Out-of-fold, always. This is the single most common way
  a CTR pipeline scores 0.75 on validation and 0.59 on hidden test.
- **Row sampling breaks GAUC.** Subsample users.
- Remember `(user_id, video_id)` is not unique — `row_id` is the key. Any join or dedupe that
  assumes uniqueness silently misaligns the submission.
