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

- [ ] `contracts.py` + `journal.py` + stubs frozen (H+2 — everyone is blocked on this)
- [ ] loop, tree, convergence, budget
- [ ] `kill -9` then `--resume` verified
- now:
- blocked on:

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

- [ ] `system.md` + `draft.md` + `improve.md` + `repair.md`
- [ ] idea bank entries: 0 / 30
- [ ] reference pipeline beats baseline
- [ ] Devpost draft
- now:
- blocked on:

## Requests

Cross-team asks. Format: `A -> C: need X because Y`.

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

Nothing proposed. Format: what changes, why, who acked, when the old shape can go.

## Fault-injection results (B fills, D uses in the writeup)

| Fault | Class | Recovered | Attempts |
|---|---|---|---|
