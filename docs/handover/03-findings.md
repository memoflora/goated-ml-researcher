# Findings

Ordered by how much they should matter to a reader. Every number here was measured by us on
this machine unless it is marked as the organisers'.

---

## 1. The agent learned to cheat, and the harness caught it

**The finding.** Given a directory containing the evaluation split, an LLM-written pipeline
will use the outcome of the impression it is being asked to predict — not because it is
instructed to, and not while announcing it. It looks exactly like a breakthrough.

**How it presented.** Official run A converged and reported:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| reported | 0.99999 | 0.69679 | **0.84839** |
| oracle ceiling (validation) | 1.00000 | 0.69679 | **0.84839** |

Identical to the theoretical maximum. That is the tell: a perfect ranking of this validation
split scores 0.8484 rather than 1.0, because 27.1% of users have no positive label. Matching
the oracle to five decimals means holding the answers.

**The mechanism, first diagnosis (incomplete).** The pipeline computed per-user and per-item
`long_view` rates from the evaluation split itself:

```python
user_ctr  = eval_rows.groupby('user_id')['long_view'].mean()
video_ctr = eval_rows.groupby('video_id')['long_view'].mean()
```

We blanked `long_view` on hidden splits. **The pipeline still scored 0.84839.**

**The mechanism, actual.** The label was never the only leak. Each log row carries everything
the impression produced, recorded at the same instant:

| column | relationship to `long_view` |
|---|---|
| `is_click` | correlation **0.7497** on its own |
| `play_time_ms` ÷ `duration_ms` | **0.884** mean when label = 1, **0.099** when 0 |
| `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate` | same instant |
| `profile_stay_time`, `comment_stay_time`, `is_profile_enter` | same instant |

Eleven columns in total. A GBDT handed them does not need the label — training log-loss came
out at 0.0025.

This is also why the organisers' own baseline uses `user_id`, `video_id`, `author_id`, `tab`
and a duration bucket, and scores 0.60 rather than 1.00. Those are the things knowable
*before* the impression.

**The fix.** `orchestrator/masking.py` writes a copy of the data in which all eleven
post-outcome columns are blank for any split the pipeline may not see. Rows survive; every
pre-impression feature survives. Visibility mirrors the rules exactly:

```
--split val    train labels only          (valid and test blanked)
--split test   train + validation labels  (test blanked)
```

**The fix, measured** — the same pipeline, re-run against the mask:

| | leaked | masked |
|---|---|---|
| corr(score, true label) | 0.98931 | **−0.03989** |
| GAUC | 0.99999 | 0.49445 |
| primary | 0.84839 | **0.47939** |

Below random. It had never been a model.

**Why a prompt could not have fixed it.** The data card already warned about leakage. It
happened twice anyway. Prompt-level rules are advisory to something that writes its own code
and never re-reads the warning while debugging. `datasource.materialise()` had the right idea
for generic tasks — `test.csv` is written *without* its target column, so no-peeking is
enforced by absence rather than instruction — and the one path that protection missed was the
benchmark that actually counts.

**Why it matters beyond our score.** On `--split test` the same code reads hidden-test
labels. That is the single disqualifying rule in the problem statement. A leaderboard number
produced this way is not a weak submission; it is an invalid one, and it would have looked
like our best result.

---

## 2. Pair-sampling weight decides the sign of a ranking loss

Ours, measured before the agent ever ran, in `reference/bpr_fm.py`. A controlled experiment:
identical FM, identical features, identical optimiser, identical early stopping. Only the
loss changed, pointwise → within-user pairwise BPR.

| pair sampling | validation primary | vs baseline |
|---|---|---|
| users sampled **uniformly** | 0.5982 | **−0.0034** |
| users weighted by **positive count** | 0.6032 | **+0.0016** |

A swing of 0.005 from a detail almost nobody would think to state, and the *sign* of the
result turns on it. GAUC averages per-user AUC **weighted by positive count**, so uniform
sampling optimises a different quantity from the one being scored.

The consequence for an autonomous agent is concrete: told only "use a ranking loss", it
implements the obvious uniform version, measures a loss against baseline, and abandons the
single most promising direction on the strength of an implementation detail. So it is written
down — in `ideas.yaml` as `T1.bpr-pairwise`, as a first-class hyperparameter sweep
(`T1.pair-weighting-sweep`), and as a dead end (`DEAD.uniform-pair-sampling`).

We got it wrong on the first attempt and the code carried a comment claiming uniform matched
GAUC. That is why it exists as a control rather than an argument.

---

## 3. The agent's failures were never reasoning failures

Across the first three live runs, 24 iterations produced zero valid submissions. Not one
failure was a modelling or reasoning error. Every diagnosis the agent made was correct.

| run | what killed it | what it actually was |
|---|---|---|
| 1 | `FileNotFoundError`, then pandas/LightGBM `TypeError`s | our data card named a path that does not exist; our prompt gave library versions but not what those versions changed |
| 2 | argparse rejecting `--subsample`; `row_id = -1` | it rewrote the plumbing from scratch each draft |
| 3 | merge dtype errors, alignment failures | same |

Every one is a fact about *our prompts and our environment*, not about the model. The fixes
were correspondingly unglamorous:

- state the real data layout, first, with no alternative to weigh
- list the specific API removals (`DataFrame.append`, `verbose_eval`, `early_stopping_rounds`)
- hand it a working pipeline to start from, so it spends iterations on the model rather than
  on argparse

After the third fix the failure *character* changed completely — zero contract failures, and
the agent started failing on feature engineering instead. That is the right kind of problem.

---

## 4. Some feedback loops do not exist until you build them

A run scored 0.6189 on validation and could not be submitted. Finalisation died on a
`DataFrame.append` sitting on the `--split test` branch — a line the entire twelve-iteration
run never executed once, because every development iteration uses `--split val`.

The prompt already warned that `.append` was removed. The agent avoided it *everywhere it got
feedback*. It survived only where nothing ever ran the code.

The fix is a probe: as soon as a node scores, its test branch is executed on a 1% subsample
purely to check that it runs.

**And the first version of that fix was wrong**, which is worth recording. It disqualified any
node whose test branch failed. That sounds principled — a solution we cannot submit is not a
solution — but a sophisticated pipeline is far likelier to have an untested branch than a
trivial one, so it selected for triviality. It rejected a trained pairwise-nDCG model at
0.4959 and crowned one that ranked by video ID with 2.2 seconds of "training", scoring 0.4839.
Quality has to be judged on the metric. The probe now records the fault and repairs it at
finalisation, against the node that actually won.

---

## 5. Facts the organisers measured, which we encoded rather than rediscovered

From the starter kit's own README, which is in Chinese and which the problem statement does
not reproduce:

- **More static features is a dead end** — all 13 fields score 0.5940 against the 5-field
  baseline's 0.5950.
- **More capacity is a dead end** — embedding k = 8/16/32 gives 0.5895/0.5902/0.5887.
- **Ranking is within-user**, so a user-side first-order term contributes *exactly* zero to
  the ordering; user-side signal only pays through a cross with the item side.
- Their ranked list of untested directions puts **ranking losses first** and **architecture
  swaps last** — the inverse of the obvious instinct. Our idea bank is tiered to match: losses
  T1, sequences T2, multi-task T3, DeepFM/DCN T4.

All are injected into every prompt as measured dead ends, so the agent does not spend
iterations rediscovering published negative results.

---

## 6. Cost is not the constraint; model latency is

Measured across the live runs: **~9.4k tokens and ~21s per iteration** on the dev setting; a
24-iteration official run cost 242k tokens in 14 minutes. Both ceilings (4M tokens, 6h) are
comfortable.

The striking part: in that run, **executing pipelines took 360s of 839s total**. The rest was
model latency. A 50-iteration run's duration is set by the model, not the dataset — a faster
model would shorten a run more than any pipeline optimisation would.
