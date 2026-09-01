# What we built

## The loop

```
        ┌──────────────── ideas.yaml + data card ────────────────┐
        ▼                                                        │
   agent.py ──proposal──▶ pipeline.py ──▶ sandbox.py ──result──▶ core.py
   (LLM writes            (one self-      (subprocess,           (tree, policy,
    a whole file)          contained       timeouts, error        convergence,
        ▲                  program)        classification)        budget)
        │                                       │                     │
        └──────── repair, on classified ────────┘                     ▼
                  failure (3 attempts)                     evaluate.py ─▶ metrics
                                                                  │
                                            journal.jsonl ◀───────┘
                                                  │
                                                  ▼
                                    report.py ─▶ RESULTS.md + trajectory.png
```

Shaped after AIDE ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)): ML engineering as
code optimisation over a tree of solution programs. The agent writes a complete `pipeline.py`
every iteration — never a diff — and the orchestrator scores its *submission*, never trusting
a number the pipeline reports about itself.

## The design decisions that mattered

**One self-contained file per solution.** The agent's entire output surface is one program
behind a frozen CLI (`--data-dir --out-dir --split --seed --subsample`). Sandboxing, diffing,
reverting and re-running all become trivial, and the agent cannot reach into our code.

**Scoring lives outside the pipeline.** The pipeline writes scores; `evaluate.py` computes
metrics, delegating every one to the organisers' vendored `evaluate.py`. That pins the
conventions and means a pipeline cannot mark its own homework.

**Frozen contracts.** `contracts.py` holds the dataclasses every module shares, frozen after
the first two hours. Four people built against them in parallel with stubs behind every seam;
the mismatches that did occur surfaced as test failures rather than as runtime surprises.

**A dataset is configuration.** `tasks/*.yaml` describes a problem — target, splits, metrics,
submission schema — and everything else is derived: the data card from an automatic profile,
the splits materialised once, the metric direction handled so the search always maximises.
A second, completely different problem (synthetic tabular regression) runs through the same
loop, which is how we know the orchestrator is not KuaiRand-shaped.

**Enforcement by absence, not instruction.** The most important lesson of the project. Rules
given to the agent in prose are advisory to something that writes its own code; rules enforced
by removing the data are not. `datasource.materialise()` writes `test.csv` without its target;
`masking.py` blanks post-outcome columns on hidden splits. See
[03-findings.md](03-findings.md) §1 for what happened before that existed.

**The journal is a deliverable, not a log.** Append-only JSONL, flushed on every write, one
object per event, with the hypothesis attached to every proposal. `RESULTS.md` and the
trajectory chart regenerate from it alone, so a run that dies at hour four still has a current
deliverable, and a reader can audit every decision the agent made.

## Robustness

Failures are classified (`syntax`, `import`, `data`, `runtime`, `oom`, `timeout`, `contract`,
`eval`) and repaired up to three times with the error excerpt and the previous attempts in
context; then the node is marked dead and the search routes around it. A fault-injection suite
covers every class end to end. Runs checkpoint atomically each iteration and resume from
`state.json`, because a crash that needs a human costs us on the Autonomy criterion.

## Providers

`agent.py` speaks the Anthropic Messages shape; OpenAI and Gemini are adapted to it, so
nothing above that line knows which is live. Failover is deliberately narrow: auth or
model-not-found disables the primary for the run, 429/5xx after backoff fails over for one
call only, and it **never** fails over on a malformed proposal — that is the repair loop's
job, and conflating the two would hide model quality problems behind a provider switch.
`summary.json` records which provider actually served, because Feasibility is scored on
tokens and a silent failover would make our reported numbers wrong.

## What is offline

`--mode smoke` pins the LLM, sandbox and evaluator to stubs: no key, no dataset, and it is
what CI runs. `--agent replay` serves canned real pipelines against the real dataset and
reaches 0.5807 for zero tokens, so the entire loop can be exercised — and reviewed — without
a key or a budget.

## Test suite

425 tests, lint clean. The ones worth knowing about:

- the loop scores a real pipeline on real data and lands a valid final submission
- item popularity reproduces to 0.58072 and the FM baseline to 0.6015 **through our own
  stack**, so we know the scoring is right rather than assuming it
- fault injection for every error class
- the pipeline whitelist must be importable and its pins must match what is installed — a
  whitelist that lies costs an iteration every time the agent believes it
- prompt-content tests: no shared prompt may name a task-specific constant, and no prompt may
  render a variable `agent.py` does not supply

That last category caught two real regressions. They are failures of *what the agent is told*,
and nothing in a conventional suite would have found them.
