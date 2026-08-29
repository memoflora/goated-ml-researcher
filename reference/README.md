# Reference pipelines — calibration, not submission

Nothing in this directory is ever submitted, and none of it is shown to the agent. These
are hand-written controls that answer questions the agent's own trajectory cannot answer
about itself:

- Is there real headroom above the baseline?
- When the agent plateaus, is the bottleneck the **agent** or the **dataset**?
- Are the idea bank's top-tier claims actually true on this data?

That last one matters most. The whole T1 tier rests on the organisers' claim that the
pointwise/ranking objective mismatch is the biggest untested win here. If that is wrong we
need to know now, not at H+40 with a scored run already committed.

## `bpr_fm.py` — does a ranking loss beat pointwise logloss?

A controlled experiment. Exactly one thing changes from the official baseline:

| | baseline | this |
|---|---|---|
| features | the 5 official fields | **same** |
| encoder | `vendor/starter_kit/data.py` | **same, untouched** |
| model | FM, k=16 | **same, copied not reimplemented** |
| optimiser | Adam, lr 1e-3, l2 1e-6 | **same** |
| early stopping | on validation primary | **same** |
| **loss** | **pointwise logloss** | **within-user pairwise BPR** |

Trains on train, selects on validation, never reads test.

```bash
python reference/bpr_fm.py --epochs 25 --pairs-per-epoch 300000
```

### Result

Validation primary, three seeds:

| | GAUC | nDCG@5 | primary | vs baseline |
|---|---|---|---|---|
| official baseline (FM, pointwise) | 0.6674 | 0.5357 | 0.6016 | — |
| **BPR-FM, GAUC-matched sampling** | 0.6698 | 0.5365 | **0.6032** | **+0.0016** |
| BPR-FM, uniform user sampling | 0.6621 | 0.5343 | 0.5982 | −0.0034 |

Seeds 0/1/2 gave 0.6033 / 0.6032 / 0.6030 — std 0.0001, tighter than the baseline's own
five-seed std of 0.0008. The gain is small but it is not noise.

### The finding that actually matters

**The pair-sampling weight mattered more than the loss function.** Same loss, same model,
same everything — only how often each user contributes a training pair:

- sample users **uniformly** → **0.5982**, worse than baseline by 0.0034
- sample users **in proportion to their positive count** → **0.6032**, better by 0.0016

A swing of 0.005 from a detail most people would not think to mention, and the sign of the
result flips on it.

The reason is that GAUC is not a uniform average. It averages per-user AUC **weighted by
positive count**, over only those users with `0 < positives < impressions`. Uniform pair
sampling therefore optimises a genuinely different quantity from the one we are scored on.
Drawing a positive row uniformly from all positives is the same thing as drawing a user in
proportion to their positive count, so the fix is one line.

I got this wrong on the first pass — the original code sampled users uniformly and carried
a comment claiming that matched GAUC. It does not. That is precisely why this experiment
exists.

### What this means for the run

1. **"Switch to a ranking loss" is underspecified as an instruction.** An agent that
   implements the obvious thing — uniform pair sampling — measures −0.0034, concludes the
   organisers' top-ranked direction is refuted, and abandons the entire T1 tier on the
   strength of an implementation detail. `ideas.yaml` now states the weighting explicitly.
2. **The headroom is real but a single change is modest.** Validation oracle is 0.8484, so
   the remaining headroom above baseline is 0.2468. This change captures about 0.6% of it.
   Real gains will come from stacking, not from one clever loss.
3. **The agent is not the bottleneck yet.** When our autonomous runs plateau near 0.603,
   that is roughly the ceiling of *this configuration*, not evidence the agent is failing.
   Anything meaningfully past it means the agent found something this control did not.
