# STATUS

Shared scratchpad. Update your section before every sync (H+6, H+12, H+24, H+36, H+42, H+48,
H+60). Bullets, not prose.

## Key numbers

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline — validation | 0.6674 | 0.5357 | **0.6016** |
| Official baseline — hidden test | 0.6610 | 0.5282 | **0.5946** |
| **Our reproduction — validation** | **0.6671** | **0.5358** | **0.6015** ✅ |
| **Our reproduction — test** | **0.6621** | **0.5286** | **0.5953** ✅ |
| Our best autonomous run (validation) | — | — | — |
| Attainable ceiling | 1.0000 | 0.7289 | 0.8645 |

Reproduced 29 Aug on Windows / Python 3.14.2 / numpy 2.4.1, `baseline.py --model fm`, seed 0,
82 s wall-clock (early stop at epoch 11). Validation matches published to 0.0001; test to 0.0007,
inside the 0.0008 five-seed std. Harness self-check passed: `--model random` gives test primary
0.4757 (published 0.4753); `--model pop` gives 0.5715 exactly. Row counts exact:
train 1,141,112 / valid 124,909 / test 170,588.

## A — ML Engineer: Orchestrator & Run

- [x] `contracts.py` + `journal.py` + stubs frozen (H+2) — **shipped, unblocking B/C/D**
- [x] loop, tree, convergence, budget guard, accounting, final submission step
- [x] `kill -9` then `--resume` verified (`tests/test_core.py`, SIGKILLs a real child process)
- [x] 49 tests green, ruff clean, 3-iteration smoke run in **0.06 s**
- now: waiting on the real seams to land, then switching `--agent/--sandbox/--evaluator` off stub
- blocked on: nothing

**Run it — works today, entirely on stubs, no dataset needed:**

```bash
python -m orchestrator.run --mode smoke      # 3 iterations, all seams stubbed
python -m orchestrator.run --mode dev        # 8 iterations, real seams if importable
python -m orchestrator.run --resume <RUN_ID> [--max-iters N] [--wall-clock 6h]
```

Seam resolution is automatic: `auto` imports your real module if it exists and exposes its
seam, and falls back to `tests/stubs/` if it does not. Nobody waits for anybody. `smoke` pins
all three to stubs on purpose, so `make check` stays green and dataset-free whatever has landed.

Stub trajectory (`tests/stubs/executor.py`) deliberately includes a syntax error and a timeout,
so the repair path and the dead-node path are exercised on every smoke run.

Verified behaviours, each with a test:
- a node failing 3 repairs is marked `dead`, a `prune`/`route_around` event is journalled, and
  the run keeps going — `test_three_failures_kill_the_node_and_the_run_routes_around_it`
- errored iterations never enter the convergence window — `test_errors_alone_never_declare_convergence`
- a buggy node can never become `best`, even carrying stale metrics — `test_a_bad_node_cannot_poison_best`
- 3 flat scored iterations trigger an explore on the second-best distinct node
- 50-iteration run completes with valid JSONL throughout
- final submission = validation-best node rerun on `--split test`, rank-averaged over 3 seeds,
  and it survives one seed failing

## B — ML Engineer: Agent Runtime & Sandbox

- [ ] sandbox runs arbitrary `pipeline.py` with timeout + process-group kill
- [ ] repair loop with error classification
- [ ] fault-injection table passing
- [ ] `make check` in CI
- now:
- blocked on:

## C — ML Researcher: Data, Metrics & Evaluation

- [x] official baseline reproduced (val 0.6015 / test 0.5953)
- [x] random / item-popularity rungs reproduce (0.4757 / 0.5715)
- [ ] data card under 3000 tokens
- [ ] `report.py` → RESULTS.md + trajectory PNG
- now:
- blocked on:

## D — ML Researcher: Method, Knowledge & Story

- [ ] `system.md` + `draft.md` + `improve.md` + `repair.md`
- [x] idea bank entries: 32 / 30 (seeded on branch `feat/d-idea-bank`, retiered per starter-kit findings)
- [ ] reference pipeline beats baseline
- [ ] Devpost draft
- now:
- blocked on:

## Requests

Cross-team asks. Format: `A -> C: need X because Y`.

- `A -> B`: need `Makefile` with `make check` = ruff + pytest + `python -m orchestrator.run
  --mode smoke`. All three pass today (0.8 s total); I have not written the Makefile because
  it is yours. Also needs a pinned root `requirements.txt` — currently unowned, and CI cannot
  install pytest without it.
- `A -> B`: `sandbox.run` is called as `run(node, split=..., seed=..., timeout_s=...,
  subsample=...)` — keyword-only after `node`, exactly the contract. It may be a module or an
  object; `run.py` duck-types both. Same for `agent.draft/improve/repair`.
- `A -> C`: `evaluate.score()` may return `primary` or just `gauc`/`ndcg@5` — the loop fills in
  the mean if it is absent. `validate()` is called on **every** scored node, not only the final
  one, and a `False` is handled as `error_class="contract"` and repaired, so please make the
  message specific enough for an LLM to act on ("row_id gap at 12" beats "invalid").
- `A -> C`: `report.py` should read the journal by `event` and ignore unknown keys. I emit a few
  fields beyond the §5 schema — `reason` (why the policy moved where it did), `context_chars`,
  `repair_attempt`, `split`, `seeds_averaged`, `components`. All additive; nothing was removed.
  The run summary is also written to `runs/<id>/summary.json` if that is easier than the journal.
- `A -> D`: `knowledge.retrieve(tried=..., best_metrics=..., budget_left=..., k=5)` is called
  once per iteration and **must not raise** — if it does, the loop journals it and continues
  with zero ideas rather than dying. `budget_left` is iterations remaining, not tokens.

## Contract change proposed

Format: what changes, why, who acked, when the old shape can go.

**A, additive only — needs one ack.** `contracts.md` §3 describes `Context` in prose ("a plain
dataclass holding…") but never spells it out, and B cannot type `draft(ctx: Context)` against
prose. `contracts.py` therefore also defines:

- `Context` — task, run_id, iteration, data_card, ideas, history, budget, library_whitelist,
  pipeline_cli, baseline_val; plus `parent_code` / `parent_metrics` / `parent_hypothesis` on
  improve, `error_class` / `error_excerpt` / `stderr_tail` / `prior_repair_plans` /
  `repair_attempt` on repair, and `draft_angle` on draft.
- `HistoryEntry` — the compact history row. Carries a truncated hypothesis and a metric delta.
  It has no `code` field and never will; that is the token-budget guarantee in a type.
- `Budget` — `iters_left`, `seconds_left`, `tokens_left` (+ used counters).
- `primary()`, `PRIMARY_PARTS`, `PIPELINE_CLI`, `EventKind` — shared constants so the CLI
  string and the metric definition exist in exactly one place.

Nothing in §2 changed shape. Ack by replying here.

## Fault-injection results (B fills, D uses in the writeup)

| Fault | Class | Recovered | Attempts |
|---|---|---|---|
