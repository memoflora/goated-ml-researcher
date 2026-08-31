# Required deliverables, mapped to artifacts

From the problem statement, §2.5. Tick these before submitting.

| # | Required | Where it is | Status |
|---|---|---|---|
| 1 | Written project description (Devpost) | `docs/devpost.md` | **draft — needs the final numbers** |
| 2 | Public code / GitHub repository | this repo | done |
| 3 | Run and iteration logs | `runs/<id>/journal.jsonl` | done |
| 3b | **Number of manual interventions** | `runs/<id>/interventions.md` | **zero, every run** |
| 4a | Final model output in the starter-kit schema | `runs/<id>/final/submission.csv` | done — 170,588 rows, validated |
| 4b | Results table: validation-best GAUC / nDCG@5 + **absolute delta over baseline** | `01-results.md`, `RESULTS.md` | **pending the gpt-5.1 run** |
| 4c | **Resource usage**: total tokens in+out, agent wall-clock, iterations used of 50 | `runs/<id>/summary.json` | done, per run |
| 4d | GPU-hours, if any GPU used | **none — CPU only** | n/a |
| — | Bonus: KuaiRand-1k / 27k | not attempted | n/a, does not reduce the Pure score |

## What to paste for 4c

Every run's `summary.json` carries these directly:

```json
{
  "iterations": 24,          // out of the 50 cap
  "tokens_in":  204766,
  "tokens_out":  37556,
  "tokens_total": 242322,    // input + output, as required
  "wall_s": 839.087,         // total agent wall-clock, seconds
  "exec_s": 360.563,         // of which was executing pipelines
  "llm_calls": 24,
  "providers": { "openai:gpt-5.1": { "calls": .., "tokens_in": .., "tokens_out": .. } }
}
```

`providers` is worth including: it shows which model actually served, and that no silent
failover to the fallback inflated or deflated the numbers.

**GPU-hours: zero.** Every run was CPU-only. The organisers note compute is deliberately not
the binding constraint here — 100 iterations of their baseline is ~28 min on one core — and
our measurements agree: pipeline execution was 360 s of an 839 s run, the rest model latency.

## Generating the results table

```bash
python -m orchestrator.report runs/<run_id>     # writes RESULTS.md + trajectory.png
```

It reads `journal.jsonl` and nothing else, so it can be regenerated at any point, including
mid-run or after a crash.

## Two things to state explicitly in the writeup

**Interventions: zero.** Autonomy is scored primarily on this. Every run went from launch to
`summary.json` unattended. Fixes between runs went into the harness and the prompts, never
into a run in flight.

**We never scored the hidden test set.** All reported numbers are validation. The guard is
mechanical — `evaluate.score(..., "test")` refuses without `ALLOW_TEST_SCORING=1` — and after
the leak audit, post-outcome columns are physically removed from any split the pipeline may
not see. Worth saying plainly, because it is the one rule that disqualifies, and because our
own agent tried twice to break it.

## Before you submit

1. Read `docs/handover/01-results.md` and use only numbers marked trustworthy. **0.6189 and
   0.8484 are both leaked and must not appear as results.**
2. Fill the final numbers into `docs/devpost.md` from `runs/r20260831-0741/summary.json`.
3. The agent-hypothesis quotes in `docs/devpost.md` are marked `TODO`; pull them verbatim from
   the winning run's journal (`event: "proposal"`, field `hypothesis`). Do not paraphrase —
   Innovation is scored on what the agent chose to try and why, and its own words are the
   evidence.
