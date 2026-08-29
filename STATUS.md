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

## Contract change proposed

Nothing proposed. Format: what changes, why, who acked, when the old shape can go.

## Fault-injection results (B fills, D uses in the writeup)

| Fault | Class | Recovered | Attempts |
|---|---|---|---|
