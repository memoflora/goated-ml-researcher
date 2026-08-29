Iteration $iteration of run $run_id. The pipeline failed to run. Fix it.

This is repair attempt $attempt of $max_attempts. After that the node is abandoned and the
run continues from elsewhere, so a fix that works matters more than a fix that is elegant.

## What went wrong

Error class: `$error_class`

```
$error_excerpt
```

Output before the failure:

```
$stdout_tail
```

## What has already been tried on this node

$previous_attempts

If a previous attempt tried to fix this and did not, do not repeat it. Something about the
diagnosis was wrong — reconsider what the error is actually telling you rather than
applying the same fix harder.

## The code that failed

```python
$parent_code
```

## What to return

**Fix the failure. Change nothing else.** This is not the iteration to improve the model,
tidy the structure, or act on a better idea you noticed while reading. A repair that also
changes the approach makes it impossible to tell whether the fix worked, and if the result
scores differently nobody will know why.

Read the error class first, because it usually names the fix:

- `syntax` — the file does not parse. Read the reported line and the one above it.
- `import` — something outside the whitelist was imported. Rewrite that part using an
  allowed library, or drop the dependency. Do not assume it will be installed.
- `contract` — it ran but broke the output contract: no `submission.csv`, a wrong header,
  the wrong row count, or a missing `RESULT_JSON` line. Re-read the contract in the system
  prompt and satisfy it literally.
- `eval` — the submission was produced but rejected: NaN or Inf scores, or rows that do not
  align with the evaluation split. Guard the score computation and preserve row order.
- `timeout` — too slow. Cut the work: fewer epochs, a smaller candidate set, a cheaper
  feature. Do not simply hope it is faster next time.
- `oom` — too much memory at once. Stream, batch, or use a narrower dtype.
- `runtime` / `data` — read the traceback and fix the specific failure it names.

Return the hypothesis (what the error was and why your change fixes it), a short plan, and
the complete corrected file.
