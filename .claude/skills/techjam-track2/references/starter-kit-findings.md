# Starter kit findings — read before writing any idea or prompt

The starter kit ships a README (in Chinese, vendored at `vendor/starter_kit/README.md`) with far more than the Lark doc. It names dead ends
the organisers already measured and ranks the directions they left open. Ignoring it wastes
iterations on things known not to work. Everything here is verified against a local run.

## Setup that works (verified 29 Aug)

```bash
cd data
curl -sL -o KuaiRand-Pure.tar.gz https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz   # -> data/KuaiRand-Pure/data  (195 MB, gitignored)
python vendor/starter_kit/baseline.py --model fm   # also: --model pop, --model random
```

Needs Python 3.9+ and numpy, nothing else. Ran fine on Windows / Python 3.14.2 / numpy 2.4.1.
The kit is vendored at `vendor/starter_kit/` and is the **sole metric authority** - never reimplement the conventions. `orchestrator/splits.py` reads `data/KuaiRand-Pure/data` by default.

## Reproduction — all three rungs match

| Model | Split | GAUC | nDCG@5 | primary | Published | Wall |
|---|---|---|---|---|---|---|
| random | test | 0.4999 | 0.4514 | **0.4757** | 0.4753 | 5 s |
| pop | test | 0.6308 | 0.5121 | **0.5715** | 0.5715 | 5 s |
| **fm** | **valid** | **0.6671** | **0.5358** | **0.6015** | 0.6016 | 82 s |
| **fm** | **test** | **0.6621** | **0.5286** | **0.5953** | 0.5946 | |

Seed 0, early stop at epoch 11. Test is within 0.0007 — inside the 0.0008 five-seed std.
Row counts exact: train 1,141,112 / valid 124,909 / test 170,588.

FM config: k=16, lr=0.001, batch=8192, max_epochs=40, patience=4, fields =
`user_id, video_id, author_id, tab, dur_bucket`.

## ⚠️ The test labels are on our disk

`baseline.py` prints test scores locally, because KuaiRand-Pure is a public dataset and
`long_view` is right there in the CSV. "Hidden test" means the organisers score their own copy —
it does **not** mean we are unable to read it.

So the rule "develop on train + validation only" is **not enforced by the environment. We
enforce it.** And an LLM writing its own pipeline code will find those labels if nothing stops
it — tuning on test is the single easiest way for the agent to look brilliant and score badly.

Concretely:

- **A:** the orchestrator runs scored iterations with `--split val` only. `--split test` is
  reachable exactly once, at the end, for the final retrain.
- **B:** the sandbox should make the test period unreadable during development — filter it at
  the data-loading boundary, or scan generated code for test-date access and classify it as a
  `contract` error. Log any attempt; it is a great Robustness anecdote if it happens.
- **D:** `system.md` must state the rule explicitly.
- **C:** never report a test number as if it were a result. We quote validation.

## Already measured — do not spend iterations here

| Tried | Result |
|---|---|
| **More static features** — all 13 CWM feature fields (`music_id`, `video_type`, `upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs 5 fields' **0.5950**. Within noise, slightly worse. |
| **More capacity** — embedding k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887. Essentially flat. |

The organisers' explanation: the `user_id × video_id` cross already absorbs most of the learnable
signal, coarse user buckets are redundant given `user_id`, and 1.14 M rows will not support more
capacity. **The bottleneck is neither features nor capacity.**

### The structural trap: user-side first-order terms are worth exactly zero

Ranking is **within-user**. Any term that is constant inside a user does not change the intra-user
order, so pure user-side first-order features contribute nothing — measured: `item_pop × user
bias` scores identically to plain `item_pop`, to the digit.

User-side features can only matter **through crosses with the item side.** An idea that adds a
user-side feature without a cross is a wasted iteration, and the metric will not even move enough
to tell you why.

## Where the headroom actually is — the organisers' ranked list

They have *not* tested these. This is the priority order D's idea bank should follow.

1. **Change the loss function.** Training is pointwise logloss; the metrics (GAUC, nDCG) are
   *ranking* metrics. Pairwise (BPR) or listwise (softmax over that user's impressions) aligns
   the objective with how we are scored. **The organisers think this is most likely to work.**
2. **User history sequences.** The current features use *no* behaviour sequence at all, yet each
   user has hundreds to thousands of train interactions. DIN / SIM-style interest modelling is
   completely untouched.
3. **Multi-task.** The logs also carry `is_click`, `is_like`, `is_follow`, `is_comment`,
   `is_forward`, `play_time_ms` as auxiliary signals for the `long_view` main task.
4. **Watch-time modelling.** CWM's censored regression — a completed play means true watch time
   was truncated by video length, so use a one-sided loss rather than squared error.
5. **Different model** — DeepFM / DCN / xDeepFM. Explicitly deprioritised **below 1–4**, because
   capacity was measured not to be the bottleneck.
6. **Time features and distribution drift** — `hourmin`, `date`, and train-vs-test drift.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a *random-exposure*
   log, 1.18 M rows, shipped in the same download and currently unused. It is an unbiased check
   on whether a model only fits biased traffic.

This ordering inverts a natural instinct — most teams will reach for DeepFM first. It is
explicitly the least promising of the five. The idea bank is tiered accordingly:
losses are T1, sequences T2, multi-task and watch-time T3, architecture swaps T4.

## Files in the kit

| | |
|---|---|
| `evaluate.py` | metrics + every pinned convention. **Do not modify.** |
| `data.py` | loading, official splits, feature encoding. `FIELDS` is where features get added. |
| `baseline.py` | the three baselines. FM is the one to beat. |
| `baseline_scores.json` | published scores, seed variance, convergence params. Use as `TaskSpec` source of truth. |
| `submit.py` | `--make` / `--check` / `--score`. |
| `ablation_features.py` | reproduces the "more features gains nothing" result. |

`evaluate.py` is fully model-agnostic — `evaluate(user_ids, labels, scores)`. Any model works, so
C's `Evaluator` should be a thin wrapper and nothing else.

## Submission ordering rule (exact)

`row_id` is 0-based and follows the row order of `data.load()[split]`: read
`log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, filter by
date, and preserve original file order. `(user_id, video_id)` is **not** a key — 3.06% of test
rows are duplicate pairs, up to 12 times over.
