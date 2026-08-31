# Run history — every live run and what it taught

Kept in full, including the results we withdrew, because the sequence is the honest account
of how the numbers were arrived at. All runs used `gpt-4o` unless stated.

## Before any live run

The harness was verified offline first, and those numbers still stand — they are measurements
of *our stack*, not of the agent:

| check | result |
|---|---|
| Official FM baseline reproduced | validation 0.6015 vs published 0.6016 |
| Item popularity through our evaluator | 0.58072 vs published 0.5807, exact |
| Random / harness self-check | 0.4757 vs published 0.4753 |
| Row counts | 1,141,112 / 124,909 / 170,588, exact |
| `reference/bpr_fm.py` control | 0.6032, +0.0016 over baseline |

## The live runs

### Runs 1–3 — 24 iterations, zero valid submissions

Each fix removed a class of failure and revealed the next.

| run | failure | fix |
|---|---|---|
| 1 | `FileNotFoundError`; then pandas `.append`, LightGBM `verbose_eval`, `early_stopping_rounds` | the data card led with a path that does not exist; the prompt gave library versions but not what those versions removed |
| 2 | argparse rejecting `--subsample`; `row_id = -1`; feature columns dropped before indexing | rewriting plumbing from scratch every draft |
| 3 | merge dtype mismatch, `df["a","b"]`, alignment after merges | gave it a working `skeleton.py` to start from, and taught that `row_id` is positional, not derived |

Not one of these 24 failures was a reasoning error. The agent's diagnoses were correct every
time. They were facts about our prompts and environment. Details in
[03-findings.md](03-findings.md) §3.

### Run 4 — 0.6189, later withdrawn

`r20260831-0532` · 12 iterations · converged · 124,243 tokens · 509 s · gpt-4o

Reported validation primary **0.6189**, +0.0173 over baseline, from a LambdaRank pairwise
model the agent chose itself. Its hypothesis, verbatim:

> *"The current model uses pointwise loss, which doesn't align well with ranking metrics like
> GAUC and nDCG@5. By switching to a pairwise loss function and weighting the pairs by the
> change in nDCG@5 they cause…"*

That reasoning is genuinely good — it is the organisers' own top-ranked untested direction.
**But the result is invalid.** The same pipeline computed CTR features from the validation
split's own labels (`compute_features(logs)` at line 102). It also produced no submittable
artifact: finalisation failed on a `DataFrame.append` on the `--split test` branch, a line
the run never executed.

**Do not report 0.6189.** The hypothesis is quotable; the number is not.

### Official run 1 — 0.4839, valid submission

`r20260831-0633` · 24 iterations · converged · 242,322 tokens · 839 s

First run to produce a valid 170,588-row submission. It scored **below random**, and the cause
was ours: a test-path check that disqualified any node whose test branch failed. Sophisticated
pipelines break there more often than trivial ones, so it selected for triviality — rejecting
a trained pairwise-nDCG model at 0.4959 and crowning one that ranked by video ID with 2.2
seconds of "training". Fixed: the probe now records the fault and repairs it at finalisation
rather than vetoing.

### Official run A — 0.8484, the leak

`r20260831-0708` · 21 iterations · converged · 228,293 tokens · 953 s

Reported primary **0.84839**, GAUC **0.99999** — exactly the oracle ceiling. This is what
uncovered the leak. Full account in [03-findings.md](03-findings.md) §1. Re-run against the
completed mask, the same pipeline scores **0.4794**, below random, with correlation to the
truth falling from 0.98931 to −0.03989.

### Official run on gpt-5.1 — the one clean result

`r20260831-0741` · 13 iterations of 50 · converged · 270,078 tokens · 6,514 s · `gpt-5.1`

The first run on the leak-proof harness, and the only agent number we can defend.

**Validation primary 0.59184** — GAUC 0.65445, nDCG@5 0.52924 — which is **0.0098 short of
the 0.6016 baseline**, and above the 0.5807 item-popularity heuristic. Trajectory:
0.58458 → 0.56031 → 0.59073 → **0.59184** → 0.59040 → 0.59184, then converged under the
ε = 0.002 / N = 3 rule. It stopped because it stopped improving, not because it ran out of
budget: 13 of 50 iterations, 270k of 4M tokens.

**No submission.** The winner's `--split test` branch died with a Windows access violation
(`0xC0000005`) — a native crash inside LightGBM on the larger train+validation fit, not a
Python error, so the repair loop had nothing textual to work with. Three repair attempts and
all three final seeds hit the same crash.

The test-path probe worked correctly: it flagged the broken branch at iterations 10, 11 and 13
and left the node eligible to win on the metric. The earlier, vetoing version of that check
would have discarded the best model. The probe is not the problem here; the segfault is, and
it is unfixed.

Note the cost profile inverted versus gpt-4o: **5,630 s of the 6,514 s was executing
pipelines**, not waiting on the model. gpt-5.1 writes heavier pipelines and the run was on
full data.

## Reading the table of contents

| run | primary | valid submission | trustworthy |
|---|---|---|---|
| r20260831-0532 | 0.6189 | no | **no — leaked** |
| r20260831-0633 | 0.4839 | yes | yes, but the search was biased |
| r20260831-0708 | 0.8484 | yes | **no — leaked** |
| r20260831-0741 | **0.5918** | no (crash) | **yes — the clean result** |

## Cost, measured

~9.4k tokens and ~21 s per iteration; a 24-iteration official run cost 242k tokens in 14
minutes. Both ceilings (4M tokens, 6 h) are comfortable. In that run, executing pipelines took
360 s of 839 s — **the rest was model latency**, so run duration is set by the model rather
than the dataset.

## Manual interventions

**Zero during any run.** Every run completed unattended, from launch to `summary.json`. The
fixes between runs were made by us to the harness and the prompts, never to a run in flight;
`runs/<id>/interventions.md` is empty in every case.
