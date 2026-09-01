# Results — `r20260901-0237`

Generated from `journal.jsonl` by `python -m orchestrator.report`.

## Headline

| metric | official baseline (val) | our validation best | absolute delta |
|---|---|---|---|
| gauc | 0.6674 | 0.6397 | **-0.0277** |
| ndcg@5 | 0.5357 | 0.5241 | **-0.0116** |
| primary | 0.6016 | 0.5819 | **-0.0197** |

Best node `n009` at iteration 10. Primary 0.5819. That captures 27.4% of the attainable range (random 0.4753 -> ceiling 0.8645); the official baseline captures 32.5%.

## Resource accounting

| | |
|---|---|
| Iterations used | 12 |
| Scored nodes | 6 |
| Total LLM tokens (in + out) | **311,992** |
| — input | 226,532 |
| — output | 85,460 |
| Agent wall-clock | 0.57 h (2,040 s) |
| GPU-hours | 0.00 (CPU only) |
| Model | `gpt-5.6-terra` |
| Converged | yes |

## Autonomy

**Manual interventions during this run: 0.**

Every human touch is logged as an `intervention` event in the journal and in
`interventions.md`. Zero is the target; each one is treated as a bug in the agent.

## Robustness

- Failed steps: **6**, of which **5** were recovered automatically.
- Nodes abandoned after the repair budget, routed around: 0.

| error class | count |
|---|---|
| runtime | 3 |
| contract | 3 |

A failed node never stops the run: it is repaired up to three times, then marked dead
and the search routes around it.

## Score trajectory

| iter | node | GAUC | nDCG@5 | primary | vs baseline |
|---|---|---|---|---|---|
| 1 | `n000` | 0.6255 | 0.5175 | 0.5715 | -0.0301 |
| 7 | `n006` | 0.6397 | 0.5240 | 0.5819 | -0.0197 |
| 9 | `n008` | 0.5939 | 0.5041 | 0.5490 | -0.0526 |
| 10 | `n009` | 0.6397 | 0.5241 | 0.5819 | -0.0197 |
| 11 | `n010` | 0.6363 | 0.5225 | 0.5794 | -0.0222 |
| 12 | `n011` | 0.6275 | 0.5201 | 0.5738 | -0.0278 |

## What the agent tried, and why

The hypothesis behind every proposal, in order. This is the Innovation record.

**1 · `n000` · draft** _(ideas: T0.reproduce-fm)_

> The official baseline should reach roughly validation primary 0.6016 because the logged impression outcome has strong sparse user-by-video, user-by-author, and context interactions that are represented directly by a k=16 factorization machine; using only the five prescribed categorical fields prevents a first draft from confusing untested side-feature effects with basic pipeline correctness.

**2 · `n001` · draft** _(ideas: T0.early-stop-on-primary, T0.item-pop-prior)_

> The main limitation of the constant floor is that it ignores the highly repeatable item- and user–item-level long-view propensity in this logged exposure data. A leakage-safe, temporally accumulated set of smoothed CTR and exposure/recency features, combined with a modest GBDT that can cross those item signals with user and context identifiers, should reproduce or modestly exceed the 0.6016 validation primary (roughly 0.60–0.62) without embedding capacity; this claim would be false if the observed item propensities do not persist from the training dates into validation.

**3 · `n002` · debug**

> The runtime failure is caused by pandas being asked to parse `long_view` as a non-nullable NumPy `int8` even though that source column contains at least one missing value; changing only that input dtype to pandas' nullable `Int8` preserves the same values and downstream feature/model logic while allowing CSV parsing to complete, so validation primary should be unchanged apart from normal run noise.

**4 · `n003` · debug**

> The validation interaction file is carrying missing `long_view` values for at least some evaluation impressions, so the failure is caused specifically by passing a nullable held-out label series to LightGBM rather than by the feature matrix. Restricting early-stopping monitoring to evaluation rows whose labels are actually present (and falling back to the existing maximum-round fit if none are present) should make validation execution succeed without changing training features, fitted training labels, prediction rows, or their order; score movement should be negligible when labels are complete and otherwise reflects only removal of unusable monitor rows.

**5 · `n004` · debug**

> The absence of a traceback combined with a completed RESULT_JSON suggests a native-library shutdown or threading failure rather than a Python/data exception; LightGBM was the only compiled training component configured for parallel execution. Forcing its OpenMP and LightGBM execution to one thread should eliminate a thread-runtime crash while preserving the identical features, data, objective, and model hyperparameters, with no expected validation-score movement beyond normal noise.

**6 · `n005` · draft** _(ideas: T1.listwise-softmax)_

> The limiting issue for a first usable KuaiRand model is that global pointwise probability fitting does not match the per-user ordering used by both GAUC and nDCG@5. I expect a deterministic LightGBM LambdaRank model trained on complete user impression lists, using the official user/video/author/tab/duration fields and early stopping on held-out nDCG@5, to reach roughly the published 0.6016 validation primary (within a few thousandths) because its loss directly rewards moving positive impressions above negatives within each user.

**7 · `n006` · debug**

> The program reached its RESULT_JSON line, so the likely failure is downstream submission validation rather than LightGBM training: `prepare_features` overwrites `evaluation['user_id']` and `evaluation['video_id']` with LightGBM category codes, and those codes are then written as the submission identifiers. Preserving the original evaluation identifiers for output while continuing to use the same encoded feature frame should make the submission align with the loaded split and raise validation primary from rejection to the model's actual score.

**8 · `n007` · improve** _(ideas: T1.bpr-pairwise)_

> The current LambdaRank tree model is likely limited by its pointwise categorical splits rather than its use of grouped rows: it cannot efficiently represent the dense user-by-video/author/context crosses that determine an impression's order. Replacing it with a compact factorization-machine scorer trained directly with positive-count-weighted within-user BPR pairs should align the loss with GAUC while learning those crosses, and I expect a material recovery toward the 0.60 primary baseline (roughly +0.01 to +0.02 if the pairwise FM transfers).

**9 · `n008` · debug**

> The runtime failure is caused by an unintended PyTorch broadcast in the FM first-order term: `squeeze(1)` does not remove the final singleton embedding dimension from a `[batch, 5, 1]` tensor, so the `[batch, 1]` linear result added to the `[batch]` interaction result becomes `[batch, batch]`. Replacing it with `squeeze(-1)` restores a one-dimensional `[batch]` model output, which should allow batched evaluation assignments to succeed without changing the model or training procedure.

**10 · `n009` · improve** _(ideas: T0.l2-and-lr-schedule)_

> The current LambdaRank model is likely overfitting sparse, high-cardinality user/video categorical partitions: its validation-selected ranking is still far below the known five-field baseline, and the supplied evidence specifically identifies regularisation as remaining hyperparameter headroom whereas embedding capacity was flat. Increasing only the tree L2 penalty should make leaf scores less sensitive to transient train-user/item outcomes and improve date-forward within-user ordering by roughly 0.002–0.006 primary if this is the binding limitation.

**11 · `n010` · improve** _(ideas: T1.pair-weighting-sweep)_

> The five-field FM is presently optimized with pointwise BCE even though both validation metrics judge only within-user order; its 0.5715 primary indicates that calibrated global long-view probabilities are not transferring to the short held-out lists. Replacing only the training objective with BPR comparisons sampled by positive impressions (thus weighting users by positive count, as GAUC does) should make the same FM allocate its capacity to user-specific order and plausibly recover several points of primary score.

**12 · `n011` · improve** _(ideas: T1.hard-negative-pairs)_

> The positive-count-weighted BPR FM is likely spending too many of its pair updates on negatives that it already separates from a user's positives, while the short-list nDCG@5 component is determined disproportionately by the few negatives receiving high scores. Replacing only the per-user uniform negative draw with a draw from each user's current top-scoring negative quartile after a warm-up epoch should focus the same model capacity on ordering-confusing impressions and raise validation primary by roughly 0.002–0.005; if it does not, hard-negative concentration is too noisy for these small user lists.

