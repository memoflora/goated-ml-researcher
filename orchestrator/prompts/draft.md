Iteration $iteration of run $run_id. This is a **cold start**: there is no parent pipeline
to build on. Write the first complete `pipeline.py`.

## Your angle for this draft

$draft_angle

Other drafts in this phase are being written from different angles. Commit to yours rather
than hedging toward a compromise none of them would have written — three genuinely
different starting points are worth more than three similar ones.

## Aim at the baseline, not past it

The first draft's job is a **correct, complete, reproducible pipeline that reaches the
official baseline**. Not a clever one. Everything after this iteration is improvement, and
improvement needs something that runs to improve on. A draft that reaches the baseline is a
success; a draft that reaches for a research idea and crashes has cost the run an iteration
and taught it nothing.

Concretely, prefer: a straightforward model over a novel one, an explicit loop over a
dense vectorised trick, deterministic seeding everywhere, and early stopping on the
**validation metric** rather than on training loss — those two diverge, and the metric is
what is scored.

## Ideas on file

$ideas

## Budget

$budget

## What to return

A hypothesis stating what you expect this pipeline to score and why, a short plan, and the
complete file. Make it run.
