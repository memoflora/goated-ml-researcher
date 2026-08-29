# Role A — Orchestrator and Search

You own the brain: the iteration loop, the solution tree, the search policy, convergence,
budget, and crash recovery. If the run stalls, crashes, or needs a human, that is your bug.

## You own

```
orchestrator/contracts.py      the frozen dataclasses (you write these FIRST, by H+2)
orchestrator/core/loop.py      the iteration loop
orchestrator/core/tree.py      solution tree, node lifecycle, best tracking
orchestrator/core/convergence.py
orchestrator/core/budget.py    iteration / wall-clock / token guards
orchestrator/core/state.py     atomic checkpoint + resume
orchestrator/core/context.py   assembles the Context handed to the agent
orchestrator/search/policy.py  draft / improve / debug selection
orchestrator/run.py            the CLI
tests/test_loop.py, tests/test_policy.py, tests/test_convergence.py
```

## You do not touch

`orchestrator/agent/`, `orchestrator/exec/` (B) · `orchestrator/eval/`,
`orchestrator/knowledge/` (C) · `orchestrator/report/`, `docs/` (D).

## First task, before anything else (due H+2)

Write `orchestrator/contracts.py` exactly as specified in `references/contracts.md`, plus
`tests/stubs/` containing a `StubAgent` (returns canned proposals from a fixture directory)
and a `StubExecutor` (returns canned `ExecResult`s, including failures). Push it and announce
in `STATUS.md`. **The other three are blocked until this lands.** Nothing else you do this
weekend matters as much as shipping this in the first two hours.

## Then, in order

1. **Loop skeleton** driving the stubs for 3 iterations, emitting journal events via D's
   `Journal` protocol (stub it until D lands the real one).
2. **Tree + best tracking.** Nodes are immutable once scored. `best` = highest validation
   primary among `status == "ok"` nodes.
3. **Convergence detector.** Converged when validation primary has not improved by more than
   `eps = 0.002` over the last `N = 3` **scored** iterations. Errored iterations do not count
   toward N. Also stop at 50 iterations or 6 h wall-clock, whichever comes first. Unit-test
   this against synthetic score sequences including plateaus, noise, and late jumps.
4. **Budget guard.** Track iterations, wall-clock and cumulative tokens. Refuse to start an
   iteration that cannot finish inside the wall-clock ceiling.
5. **Checkpoint and resume.** Rewrite `state.json` atomically (write temp, fsync, rename)
   every iteration. `--resume <run_id>` rebuilds the tree and continues. **Test this by
   `kill -9`ing a run mid-iteration.** Crash recovery without a human is worth real points.
6. **Search policy** (see `references/architecture.md` for the rules): draft phase, greedy
   improve, debug-first, explore on stall, anti-overfit tie-break toward the simpler node.
7. **Context assembly.** This is where token cost is won or lost. The context carries: task
   card, C's data card, the parent's full code, the parent's metrics, top-K ideas from C,
   and a **compact** history — hypothesis plus metric delta only, never full code of past
   attempts. Cap it and log the size.
8. **Final submission step.** At convergence: take the validation-best node, rerun it with
   `--split test` (trains on train + validation), average over 3 seeds if time allows, run
   C's validator, and write `final/submission.csv`. Emit `converged` and `run_end` events.

## Your CLI

```
python -m orchestrator.run --task kuairand-pure --mode {smoke,dev,official} \
       [--max-iters N] [--wall-clock 6h] [--resume RUN_ID] [--seed N]
```

Default is fully unattended. There is no interactive mode. If you find yourself wanting one,
that is a signal to make the agent handle the case instead.

## Acceptance tests — you are done when

- [ ] `make check` runs `smoke` (3 iterations, stubbed LLM) in under 60 s, green
- [ ] 50-iteration dry run with stubs completes, journal is valid JSONL throughout
- [ ] Convergence detector passes unit tests on plateau / noise / late-jump sequences
- [ ] `kill -9` mid-iteration, then `--resume`, continues correctly with no lost nodes
- [ ] A node that fails 3 times is marked `dead` and the run routes around it, logging a
      `recovery` event with `"route_around"`
- [ ] Three consecutive non-improving iterations trigger an explore action
- [ ] `official` mode run reaches `converged` or the 50/6h cap with **zero** interventions

## Traps

- **Do not let a bad node poison `best`.** Only `status == "ok"` nodes with a validated
  submission are eligible.
- **Do not count errored iterations toward convergence.** A run that errors 3 times in a row
  would otherwise declare victory at iteration 4.
- **Do not select the final submission on a single seed.** Baseline seed noise is 0.0008 and
  the convergence eps is 0.002; a 0.001 "win" is nothing.
- **Do not add human prompts anywhere.** Autonomy is directly scored.
- Keep `state.json` small. If it grows past a megabyte you are storing code in it — store
  paths instead.
