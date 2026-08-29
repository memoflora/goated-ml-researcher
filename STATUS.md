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

**Independently confirmed on macOS 15 / Python 3.13.6 / numpy 2.3.5 (C, H+10):** identical to
four decimals on every rung — val 0.6015, test 0.5953, pop 0.5807/0.5715, random 0.4827/0.4757,
same early stop at epoch 11, 25 s wall-clock. Two OSes, two Python versions, same numbers, so
the reproduction is a property of the harness and not of one machine.

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

- [x] official baseline reproduced (val 0.6015 / test 0.5953), confirmed on a 2nd OS
- [x] random / item-popularity rungs reproduce (0.4757 / 0.5715)
- [x] data card — **~1435 tokens**, well under the 3000 budget, generated from the data
- [x] `report.py` → RESULTS.md + trajectory PNG, works on a fixture journal
- [x] `evaluate.py` — `score()` / `validate()`, agrees with `evaluate.py` exactly
- [x] `splits.py` — cached split loader, 3.7 s cold → 0.085 s warm
- [x] `requirements-pipeline.txt` — the import whitelist for generated code
- [x] 39 unit tests, 1.0 s
- now: idle on my critical path — everything C owns for Phase 0/1 is landed and green.
- blocked on: nothing. Next up unless someone needs otherwise: seed-averaged scoring helper
  for A's final selection, and a `--subsample` user-sampling helper the agent can copy.

**What landed (all under `orchestrator/`, flat layout per the new contracts):**

| file | what |
|---|---|
| `splits.py` | cached `(user_id, video_id, long_view)` per split in `row_id` order |
| `evaluate.py` | `score(sub, split)`, `validate(sub, split)`, `delta_vs_baseline()` |
| `datacard.py` | `data_card()` — the EDA summary the LLM reads |
| `report.py` | `python -m orchestrator.report runs/<id>` → RESULTS.md + trajectory.png |
| `vendor/starter_kit/` | the kit, **unmodified**. `evaluate.py` is the metric authority. |
| `tests/fixtures/journal_sample.jsonl` | 19-event journal + 1 malformed line, for D and me |

Notes for whoever integrates:

- `score()` / `validate()` accept **either** `"val"` or `"valid"`. The pipeline CLI contract
  says `val`, the starter kit says `valid`; both work everywhere.
- Metric keys are lowercase `gauc` / `ndcg@5` / `primary`, matching the journal schema.
- **`score(..., "test")` refuses to run** unless `ALLOW_TEST_SCORING=1`. KuaiRand ships the
  test labels, so nothing technically stops us touching them; this makes the rule mechanical
  instead of a promise. The final submission needs `validate()` only, which reads no labels.
- `report.py` reads the journal and nothing else, so RESULTS.md regenerates at any moment,
  including mid-run and after a crash. It skips malformed lines and counts them.

## D — ML Researcher: Method, Knowledge & Story

- [x] `system.md` + `draft.md` + `improve.md` + `repair.md` — written to B's exact template
      variable contract; a test asserts no prompt uses a variable `agent.py` does not supply
- [x] idea bank entries: 32 / 30, retiered per starter-kit findings
- [x] `knowledge.py` — module-level `retrieve()` / `dead_ends()`, wired and feeding the loop
- [x] reference pipeline beats baseline — BPR-FM, validation primary **0.6032** vs
      baseline 0.6016 (+0.0016), seeds 0/1/2 = 0.6033/0.6032/0.6030, std 0.0001
- [ ] Devpost draft
- now: prompts + knowledge + reference pipeline landed, 35 D tests, 122 green overall
- blocked on: nothing

**D -> everyone: the pair-sampling weight matters more than the loss function.**
`reference/bpr_fm.py` is a controlled experiment — same 5 fields, same FM, same Adam, same
early stopping, only pointwise logloss -> within-user pairwise BPR. Result depends entirely
on one detail nobody would think to state:

| pair sampling | validation primary | vs baseline |
|---|---|---|
| users uniformly | 0.5982 | **-0.0034** |
| users weighted by positive count | 0.6032 | **+0.0016** |

GAUC averages per-user AUC *weighted by positive count*, so uniform sampling optimises a
different quantity from the one we are scored on. A 0.005 swing, and the sign of the result
flips on it. I had it wrong on the first pass.

Why this matters for the run: an agent that implements the obvious thing — uniform pairs —
measures -0.0034, concludes the organisers' top-ranked direction is refuted, and abandons
the whole T1 tier because of an implementation detail. `ideas.yaml` and `system.md` now
state the weighting explicitly, and it is recorded as a dead end.

Calibration read: headroom above baseline on validation is 0.2468 (oracle 0.8484). One
ranking-loss change captures ~0.6% of it. **Gains will come from stacking, not one clever
idea.** And when our autonomous runs plateau near 0.603, that is this configuration's
ceiling rather than evidence the agent is failing — anything meaningfully past it means the
agent found something this control did not.
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

**C -> D (high value, please read before writing `ideas.yaml`).** The starter kit's own
`README.md` — now vendored at `vendor/starter_kit/README.md`, it is in Chinese — has a section
the problem digest does not: the organisers list what **they already tested and found dead**,
and rank the directions they left open. This should shape the idea bank's tiers directly.

*Already measured by the organisers as no-gain — we should not spend iterations rediscovering:*

- **More static features.** Wiring in all 13 CWM feature fields (`music_id`, `video_type`,
  `upload_type` + 6 coarse user-side buckets) scored primary 0.5940 vs 0.5950 for the 5 fields.
  No difference, slightly worse. `ablation_features.py` reproduces this.
- **More capacity.** Embedding dim k = 8 / 16 / 32 → 0.5895 / 0.5902 / 0.5887. Flat.
- Their stated reason: `user_id × video_id` crossing already absorbs most of the learnable
  signal, and 1.14M rows will not support more capacity. **The bottleneck is neither features
  nor capacity.**
- **Pure user-side first-order terms contribute exactly zero.** Ranking is *within-user*, so any
  term constant across a user's rows cannot reorder that user's list — they verified `item_pop ×
  user_bias` scores identically to bare `item_pop`, to the digit. User features can only pay
  through **crosses with item-side or context features.** This is in the data card as a fact.

*Left open by the organisers, in their own order of expected payoff:*

1. **Loss function.** Training is pointwise logloss while the metrics are ranking metrics.
   Pairwise (BPR) or within-user listwise softmax aligns the objective with the scoring.
   They call this the most likely to work, and nobody has tried it.
2. **User behaviour sequences.** Completely unused today; each user has hundreds to thousands
   of train interactions. DIN / SIM-style interest modelling is a blank field.
3. **Multi-task.** `is_click` (rate 0.446), `is_like` (0.018), `is_follow`, `is_comment`,
   `is_forward`, `play_time_ms` as auxiliary heads for the `long_view` main task. All 12
   signals with their per-split rates are in the data card.
4. **Watch-time modelling / duration debiasing.** CWM's actual contribution: censored
   regression, because play time is truncated when a video ends. The data card shows the bias
   directly — mean `play_time/duration` falls 0.57 → 0.14 across duration deciles.
5. **Model swap** (DeepFM / DCN / xDeepFM) — they explicitly rank this *below* 1–4, since
   capacity is measured not to be the bottleneck. Worth reflecting in the tiering: our role
   file's suggested T2 "stronger models" is, by the organisers' own data, lower-yield than
   the loss-function and sequence work.
6. **Time features and drift.** `hourmin`, `date`, train→test drift.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a 1.18M-row
   *uniform random exposure* log covering the valid+test dates. It is part of KuaiRand, so it
   is legal, and it is a clean check on whether a model only wins on biased traffic. Note it
   is **not** in the official splits — it is not extra training data for the scored task, and
   using it as such would need care.

**C -> B: please build the sandbox venv once per run and reuse it across nodes.**
`requirements-pipeline.txt` includes torch (~100 MB+). Rebuilding per node would dominate
wall-clock, which is a scored quantity. The whitelist is deliberately tight for this reason;
it is stated in the file's header so it can be pasted into `system.md` verbatim.

**C -> A: two shapes I can provide on request** — say the word and I will write them.
(a) seed-averaged scoring for final selection, so we do not pick a node on one lucky seed
(risk register, "agent overfits validation"); (b) a user-level `--subsample` helper, since
row-level sampling silently breaks GAUC and generated code will get this wrong by default.

**C -> everyone: `requirements.txt` (tooling) does not exist yet and CI will need it.**
I installed, and my modules import: `numpy==2.3.5`, `pandas==2.3.3`, `matplotlib` (optional —
`report.py` degrades gracefully without it), `pytest`. Whoever owns repo/CI should pin these;
I did not create the file because it is shared surface, not mine.

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
