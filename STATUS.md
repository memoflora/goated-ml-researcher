# STATUS

Shared scratchpad. Everyone updates their own section before each sync (H+6, H+12, H+24,
H+36, H+42, H+48, H+60). Keep it short — bullets, not prose.

## Key numbers

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline — validation | 0.6674 | 0.5357 | **0.6016** |
| Official baseline — hidden test | 0.6610 | 0.5282 | **0.5946** |
| Our reproduction (C to fill) | — | — | — |
| Our best autonomous run (validation) | — | — | — |
| Attainable ceiling | 1.0000 | 0.7289 | 0.8645 |

## A — Orchestrator & Search

- [ ] `contracts.py` frozen (H+2)
- current:
- blocked on:

## B — Agent Runtime & Sandbox

- [ ] sandbox runs arbitrary pipeline.py with timeout
- [ ] fault-injection table passing
- current:
- blocked on:

## C — ML Core & Knowledge

- [ ] baseline reproduced
- [ ] data card
- [ ] idea bank entries: 0
- current:
- blocked on:

## D — Telemetry & Reporting

- [ ] journal writer (H+3 — A is waiting)
- [ ] dashboard on fixture
- [ ] Devpost draft
- current:
- blocked on:

## Requests

Cross-team asks go here. Format: `A -> B: need X because Y`.

## Contract change proposed

Nothing proposed. Format: what changes, why, who acked, when the old shape can be removed.

## Fault-injection results (B fills, D uses in the writeup)

| Fault | Class | Recovered | Attempts |
|---|---|---|---|
