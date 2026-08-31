# Results — with provenance

**Read the trustworthy column before quoting anything.** Two of our headline numbers were
leakage and must not appear in the writeup.

All figures are **validation**. We have never scored the hidden test set; the guard is
mechanical (`ALLOW_TEST_SCORING`), and after the leak audit the columns needed to cheat are
physically absent from any split the pipeline may not see.

---

## The benchmark ladder

Published by the organisers, and reproduced by us where marked.

| | GAUC | nDCG@5 | primary | ours |
|---|---|---|---|---|
| Random | 0.4993 | 0.4675 | 0.4834 | 0.4757 (test) ✓ |
| Item popularity | 0.6387 | 0.5227 | 0.5807 | **0.58072 ✓ exact** |
| **Official FM baseline** | **0.6674** | **0.5357** | **0.6016** | **0.6015 ✓** |
| Oracle — a *perfect* ranking | 1.0000 | 0.6968 | 0.8484 | — |

The oracle is 0.8484, not 1.0: 27.1% of users have no positive label, so their nDCG is 0 for
any model. **Judge progress against 0.8484.** Headroom above the baseline is 0.2468.

Our reproductions are the evidence that our scoring stack is correct — item popularity matches
to five decimals. The FM baseline lands 0.00013 low, and we chased that rather than waving at
tolerance: running the organisers' own untouched `baseline.py` on this machine gives the same
0.601470, so the reproduction is bit-faithful and the difference is their environment.

---

## Our own results

| result | primary | vs baseline | valid submission | **trustworthy** |
|---|---|---|---|---|
| `reference/bpr_fm.py` — hand-written control | 0.6032 | +0.0016 | n/a | **yes** |
| run `r20260831-0532`, gpt-4o | 0.6189 | +0.0173 | no | **NO — leaked** |
| run `r20260831-0633`, gpt-4o | 0.4839 | −0.1177 | yes | yes, but search was biased |
| run `r20260831-0708`, gpt-4o | 0.8484 | +0.2468 | yes | **NO — leaked** |
| run `r20260831-0741`, **gpt-5.1** | **0.5918** | **−0.0098** | no (crash) | **YES — first clean result** |

### Why 0.6189 and 0.8484 must not be reported

Both pipelines read the outcome of the impression they were predicting. The 0.8484 matched the
oracle ceiling to five decimals — the tell. Re-run against the completed mask, that same
pipeline scores **0.4794**, below random, with correlation to the truth falling from 0.98931
to −0.03989. Full account in [03-findings.md](03-findings.md) §1.

The *reasoning* in run `r20260831-0532` is still quotable and still good — it identified the
pointwise/ranking objective mismatch, which is the organisers' own top-ranked untested
direction. Quote the hypothesis; do not quote the score.

### What is safely claimable today

- The harness reproduces the benchmark ladder exactly, so our measurements are trustworthy.
- A hand-written single-change control (`reference/bpr_fm.py`) beats the baseline by +0.0016,
  and established that **pair-sampling weight decides the sign** of a ranking loss.
- The agent runs unattended to convergence with **zero manual interventions** and recovers
  from every injected fault class. A schema-valid 170,588-row submission has been produced
  (run `r20260831-0633`), so the submission path works — just not yet in the same run as a
  good score.
- **The agent has not beaten the baseline on a clean harness.** Its one defensible result is
  0.5918, which is 0.0098 short. Every number that exceeded 0.6016 was leakage.

---

## The one clean result — `r20260831-0741`, gpt-5.1

The first run on the leak-proof harness, and therefore the only agent number we can defend.

| field | value |
|---|---|
| model | `gpt-5.1` (16 calls, no fallback) |
| stop reason | **converged** at iteration **13 of 50** |
| best node | `n009` |
| validation GAUC | **0.65445** (baseline 0.6674, **−0.0130**) |
| validation nDCG@5 | **0.52924** (baseline 0.5357, **−0.0065**) |
| **validation primary** | **0.59184** (baseline 0.6016, **−0.0098**) |
| tokens in / out / total | 181,363 / 88,715 / **270,078** |
| agent wall-clock | **6,514 s** (1 h 49 m), of which **5,630 s executing pipelines** |
| GPU-hours | **0 — CPU only** |
| manual interventions | **0** |
| final submission | **none — the final-seed runs crashed** |

**It did not beat the baseline.** It converged 0.0098 short, having climbed
0.58458 → 0.56031 → 0.59073 → **0.59184** → 0.59040 → 0.59184 and then stopped improving by
more than ε = 0.002 over three scored iterations. That is the convergence rule working, not a
budget running out: it used 13 of 50 iterations and 270k of a 4M token budget.

**No submission was produced.** The winner's `--split test` branch crashed with a Windows
access violation (`0xC0000005`) — a native segfault inside LightGBM on the larger train+valid
fit, not a Python error. The repair loop tried three times and got the same crash each time;
all three final seeds then failed the same way.

Worth noting what *did* work here: the test-path probe fired correctly at iterations 10, 11
and 13, recorded the branch as broken, and — after the fix — left the node eligible to win on
the metric anyway. The earlier version of that check would have thrown away the best model.
The probe did its job; the underlying crash is a separate, unfixed problem.

### How this compares

| | primary | notes |
|---|---|---|
| item popularity | 0.5807 | a 20-line heuristic |
| **gpt-5.1 agent, clean** | **0.5918** | beats the heuristic, short of the baseline |
| official FM baseline | 0.6016 | the bar |
| our hand-written BPR control | 0.6032 | one human change |

The agent beat a trivial heuristic and fell short of a tuned baseline. That is the honest
position, and it is what should go in the writeup.

---

## Still to fill in

If another clean run is completed, record it here in the same shape.
