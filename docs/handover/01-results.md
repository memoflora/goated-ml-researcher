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
| run `r20260831-0741`, gpt-5.1 | *pending* | *pending* | *pending* | **yes — masked harness** |

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
- The agent runs unattended to convergence with **zero manual interventions**, recovers from
  every injected fault class, and produces a schema-valid 170,588-row submission.
- **We have not yet demonstrated a legitimate score above 0.6016 from the agent.** Every
  number that did so was leakage. The gpt-5.1 run is the first honest attempt.

---

## Fill this in before submitting

From `runs/r20260831-0741/summary.json`:

| field | value |
|---|---|
| model | `gpt-5.1` |
| iterations used (of 50) | |
| best validation GAUC | |
| best validation nDCG@5 | |
| best validation primary | |
| **absolute delta vs 0.6016** | |
| tokens in / out / total | |
| agent wall-clock (s) | |
| GPU-hours | **0 — CPU only** |
| manual interventions | **0** |
| final submission valid | |

Trajectory so far: iteration 1 → 0.49301, iteration 4 → 0.58458.

Then regenerate the formatted table:

```bash
python -m orchestrator.report runs/r20260831-0741
```
