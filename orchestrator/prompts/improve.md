Iteration $iteration of run $run_id. You are improving node `$parent_node_id`, which is the
best-scoring pipeline so far.

## Where it stands

$parent_metrics

## What has been tried, and what it did

$history

Read this before proposing. If a direction has already been tried and moved nothing, that
is information — either the idea is wrong on this dataset, or the implementation was. Say
which you think it is. Repeating a change that has already failed wastes the iteration.

Pay attention to the *size* of past deltas, not just their sign. Retraining an unchanged
pipeline under a different seed moves the score on its own; the task card states that
run-to-run noise figure for this dataset where it is known. A delta smaller than it is not
evidence in either direction — do not build on one as though it were established, and do not
abandon a direction on one either.

## Ideas on file

$ideas

These are ranked by expected payoff for this dataset, cheapest tier first. You are not
required to use one — propose your own if the trajectory points somewhere better, and say
what pointed you there.

## Budget

$budget

If the budget is nearly spent, prefer a change you are confident will land over a
speculative one. A modest gain that finishes beats an ambitious one that does not.

## The current pipeline

```python
$parent_code
```

## What to return

**One focused change**, with the hypothesis first: what you believe is limiting this
pipeline now, what you are changing, and what you expect it to do to the score. Then the
complete new file — the whole thing, not a diff.

Changing one thing at a time is not caution, it is measurement. Four changes at once and a
+0.004 result tells you nothing about which of the four to keep.
