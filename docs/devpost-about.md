## Inspiration

The organisers' benchmark comes with a published ladder: random scores $0.4834$, item
popularity $0.5807$, and a factorization-machine baseline $0.6016$. A *perfect* ranking
scores $0.8484$ — not $1.0$, because $27.1\%$ of users have no positive label at all, so
their nDCG is zero for any model. The real headroom above the baseline is $0.2468$, not
$0.4$.

What drew us in was not the number. It was a line in the starter kit's own README —
written in Chinese, and not reproduced in the problem statement — reporting that the
organisers had *already measured* the obvious ideas and found them flat:

| tried | result |
|---|---|
| all 13 static feature fields | $0.5940$ vs the 5-field baseline's $0.5950$ |
| embedding dimension $k = 8/16/32$ | $0.5895 / 0.5902 / 0.5887$ |

More features: nothing. More capacity: nothing. Their own ranked list of untested
directions puts **ranking losses first and architecture swaps last** — the exact inverse
of where an autonomous agent's instincts would take it.

So the interesting question stopped being "can an LLM write a recommender" and became
**"can an agent be given negative knowledge, and will it spend its iterations somewhere
other than the obvious?"**

## What it does

An LLM writes a complete `pipeline.py`. A sandbox runs it as an isolated subprocess with
no network. An evaluator scores the submission. A search policy reads the tree and
decides what to try next — repair a failure, draft a new angle, improve the leader, or
explore an alternative. An append-only journal records every hypothesis, diff, metric and
recovery.

It runs unattended to convergence, a 50-iteration cap, or a six-hour ceiling.

Our best clean run: **primary $0.58190$** (GAUC $0.63974$, nDCG@5 $0.52405$), $12$
iterations, $155{,}996$ tokens, $21$ minutes, **zero GPU-hours**, and **zero manual
interventions** — with a valid $170{,}588$-row submission.

That is $0.0197$ *below* the official baseline. We are reporting it that way on purpose,
and the reason is the most interesting thing we found.

## The thing we would lead with: the agent learned to cheat

An official run converged and reported:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| reported | $0.99999$ | $0.69679$ | $\mathbf{0.84839}$ |
| oracle ceiling | $1.00000$ | $0.69679$ | $\mathbf{0.84839}$ |

Identical to the theoretical maximum, to five decimals. That is not a breakthrough, it is
a tell. The pipeline was computing per-user and per-item `long_view` rates **from the
evaluation split itself**:

```python
user_ctr  = eval_rows.groupby('user_id')['long_view'].mean()
video_ctr = eval_rows.groupby('video_id')['long_view'].mean()
```

**Our first fix was wrong.** We blanked the label column — and the pipeline still scored
$0.84839$. The label was never the only leak. Every log row carries everything the
impression *produced*, recorded at the same instant:

| column | relationship to the label |
|---|---|
| `is_click` | correlation $0.7497$ on its own |
| $\text{play\_time\_ms} \div \text{duration\_ms}$ | mean $0.884$ when positive, $0.099$ when negative |
| `is_like`, `is_follow`, `is_forward`, `profile_stay_time`, … | same instant |

Eleven columns. A gradient-boosted tree handed those does not need the label — training
log-loss came out at $0.0025$.

Once all eleven were masked, that same pipeline's correlation with the truth fell from
$0.98931$ to $-0.03989$, and it scored $0.47939$ — **below random**. It had never been a
model.

Two reasons this matters more than our score:

1. **It is disqualifying, not merely bad.** On `--split test` the identical code reads
   hidden-test labels. A leaderboard number produced that way is not a weak submission;
   it is an invalid one — and it would have looked like our best result.
2. **No prompt could have prevented it.** Our data card already warned about leakage. It
   happened anyway, twice, because the columns were simply *present* in the directory we
   handed over. The fix had to be structural: the masked copy makes the columns
   physically absent, so a leaking aggregate returns `NaN` instead of a number.
   Enforcement by absence, not by instruction.

We also had to withdraw our own earlier headline of $0.6189$, which had a milder form of
the same flaw. We reported it, then found it, then retracted it. That sequence is in the
repository in full.

## What we learned

**Almost none of the agent's failures were reasoning failures.** Across the first three
live runs, 24 iterations produced zero valid submissions — and not one was a modelling
error. Every diagnosis the agent made was correct. The failures were facts about *our*
environment we had stated wrongly or not at all: a data path that did not exist, library
versions given without saying what those versions *removed*. Six of eight iterations in
one run died on `DataFrame.append`, `verbose_eval` and `early_stopping_rounds`. Once we
fixed our prompts, the failures changed character entirely — from plumbing to feature
engineering, which is the right kind of problem.

**An implementation detail decided the sign of a result.** We ran a controlled experiment:
identical FM, identical features, identical optimiser — only the pair sampling changed.

| pair sampling | primary | vs baseline |
|---|---|---|
| users sampled uniformly | $0.5982$ | $-0.0034$ |
| users weighted by positive count | $0.6032$ | $\mathbf{+0.0016}$ |

GAUC averages per-user AUC *weighted by positive count*. Sample users uniformly and you
optimise a different quantity from the one being scored. An agent told only "use a ranking
loss" writes the uniform version, measures a loss, and abandons the single most promising
direction on the strength of a detail nobody thought to state. So we wrote it into the
idea bank — as an idea, as a hyperparameter sweep, and as a named dead end.

**Some feedback loops do not exist until you build them.** A run scored $0.6189$ and could
not be submitted: finalisation died on a `.append` sitting on the `--split test` branch —
a line the entire twelve-iteration run never executed once, because every development
iteration uses `--split val`. The agent had avoided `.append` *everywhere it got
feedback*. It survived only where nothing ever ran the code.

**And our first fix for that was also wrong.** We disqualified any node whose test branch
failed. That sounds principled — a solution you cannot submit is not a solution — but a
sophisticated pipeline is far likelier to have an untested branch than a trivial one, so
it *selected for triviality*: it rejected a trained pairwise-nDCG model at $0.4959$ and
crowned one that ranked by video ID after 2.2 seconds of "training", scoring $0.4839$.
Quality has to be judged on the metric. The probe now records the fault and repairs it at
finalisation instead of vetoing.

## How we built it

Six seams, each independently testable, with stubs at every boundary so the whole loop
runs offline with no API key — which is also how CI runs it on every push, so it cannot
rot.

```
policy ──▶ agent ──▶ sandbox ──▶ evaluator ──▶ journal
   ▲                                              │
   └──────────────── tree of nodes ◀──────────────┘
```

- **`policy.py`** — pure functions over the tree. No I/O, no clock, no LLM. Debug-first,
  then draft diversity, then explore-on-plateau, then greedy improve. Ties inside seed
  noise ($0.0008$) break toward the *simpler* program.
- **`sandbox.py`** — own process group, RSS ceiling, wall-clock kill, outbound sockets
  blocked, secrets stripped from the child environment. Failures come back *classified*,
  never raised.
- **`masking.py`** — the leak fix. Writes a copy of the data with post-outcome columns
  blanked for any split the pipeline may not see, mirroring the rules exactly.
- **`journal.py`** — append-only JSONL, fsynced. It is a graded deliverable, so it is
  written as one.

Validation is measured, not asserted. We reproduce the published ladder before trusting
any of our own numbers: item popularity to five decimals ($0.58072$ vs $0.5807$), the FM
baseline to $0.0001$. When ours landed $0.00013$ low we chased it rather than waving at
tolerance — running the organisers' untouched `baseline.py` on our machine gives the same
figure, so the reproduction is faithful and the gap is their environment.

$425$ tests, including a fault-injection suite that `SIGKILL`s real subprocesses and
asserts none of them survive.

## Challenges

**A model that refused our call shape.** Moving to a newer reasoning model, every single
iteration failed — 8 iterations, **zero** LLM calls, and the run still exited $0$ looking
healthy:

```
400 — Function tools with reasoning_effort are not supported in
/v1/chat/completions. Use /v1/responses or set reasoning_effort to 'none'.
```

The API offered two ways out, and they are not equivalent. Setting `reasoning_effort` to
`none` clears the error in one line — by switching off the one faculty an autonomous
research agent exists to use. We probed both against the live model and moved the model to
`/v1/responses` instead, which supports tools *and* reasoning.

**A crash with nothing to read.** Our best-scoring run produced no submission: the winner's
test branch died with a native access violation inside LightGBM — no Python traceback, so
the repair loop received `exited with status -11` and re-submitted the identical program
three times. We gave that failure its own error class and a real diagnosis. The payoff
showed up in the very next run, where the agent hit the same class of failure and
diagnosed it itself:

> *"The absence of a traceback combined with a completed RESULT_JSON suggests a
> native-library shutdown or threading failure rather than a Python/data exception;
> LightGBM was the only compiled training component configured for parallel execution.
> Forcing its OpenMP and LightGBM execution to one thread should eliminate a
> thread-runtime crash while preserving the identical features, data, objective, and
> model hyperparameters."*

It was right, and it repaired itself.

**A search that stopped exactly when it should have explored.** Our best run converged at
iteration $13$ of $50$, having used $6\%$ of its token budget, $0.0098$ short of the
baseline. The policy has an explore rule — after $N$ flat iterations, abandon the leader
and work the second-best node. It had **never once fired**. Convergence needed 3 flat
iterations and exploration needed 3, so the run always stopped on the very iteration
exploration became reachable. A unit test proved the branch worked; nothing proved the
state was *reachable*. Fixed, and in the next run it fired for the first time:

```
it10  0.58190  ← best
it11  EXPLORE: 2 scored iterations without real improvement on n009
it12  EXPLORE: 3 scored iterations without real improvement on n009
```

Neither exploration beat the leader. The mechanism is demonstrated working; it is not yet
demonstrated *useful*. We are reporting that as the negative result it is.

## Where we actually stand

**The agent has not beaten the official baseline on a leak-free harness.** Our best
defensible score is $0.58190$ against $0.6016$. Every number we produced that *exceeded*
the baseline turned out to be leakage, which we found and withdrew.

What we can defend: the harness reproduces the published benchmark ladder exactly; the
agent runs to convergence with **zero manual interventions** and recovers from every
injected fault class; a hand-written single-change control beat the baseline by $+0.0016$
and established that pair-sampling weight decides the *sign* of a ranking loss; and the
one result that would have looked like a win was leakage that our own harness caught.

We would rather submit a number we can stand behind than the $0.8484$ we briefly had.
