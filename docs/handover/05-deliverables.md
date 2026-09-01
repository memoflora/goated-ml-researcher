# Required deliverables, mapped to artifacts

Every requirement the organisers list, mapped to the artifact that satisfies it. Tick
these before submitting.

## 1 · Written project description (Devpost)

| Required | Where it is | Status |
|---|---|---|
| How the solution addresses the problem statement | `docs/devpost.md` — "What it is", "The loop" | done |
| **Development tools** (VSCode, Colab, Jupyter…) | `docs/devpost.md` — "Built with → Development tools" | done |
| **APIs used** | "Built with → APIs used" — OpenAI (gpt-5.6-terra / 5.1 / 4o), Anthropic, Gemini fallback | done |
| **Libraries and frameworks** | "Built with → Libraries and frameworks" — split into orchestrator deps and the sandbox whitelist | done |
| **Datasets and assets** | "Built with → Datasets and assets" — KuaiRand-Pure, the vendored starter kit, a synthetic fixture | done |
| Final numbers and agent-hypothesis quotes | `docs/devpost.md` | **TODO — needs the run** |

## 2 · Public code / GitHub repository

| Required | Where it is | Status |
|---|---|---|
| Well-structured, commented code, all components | `orchestrator/` (~8k lines), 425 tests, ruff clean | done |
| README: project overview | `README.md` — top, "Where we stand" | done |
| README: setup and installation | `README.md` — "Quickstart", "Running it for real" | done |
| README: steps to reproduce results | `README.md` — "Quickstart" (no key needed) and "Running it for real" | done |
| README: **limitations and what you would improve** | `README.md` — "Limitations, and what we would do with more time" | done |
| README: **team member contributions** | `README.md` — "Team member contributions" (per-file ownership table) | done |

## 3 · Run and iteration logs

The Starter Kit asks for four things **per iteration**. Three come from the journal; the
code diff exists nowhere until `orchestrator/runlog.py` reconstructs it from the node
workspaces, because the journal deliberately never stores code.

```bash
python -m orchestrator.runlog runs/<run_id>     # writes RUNLOG.md
```

| Required | Where it is | Status |
|---|---|---|
| Hypothesis — what the agent intended and why | `RUNLOG.md`, from journal `proposal.hypothesis` | done |
| **The code diff applied** | `RUNLOG.md`, diffed from `nodes/<id>/pipeline.py` vs its parent | done |
| Resulting metrics (GAUC / nDCG@5) | `RUNLOG.md`, from journal `eval.metrics`, with delta vs baseline | done |
| Errors and recovery events, and how they were handled | `RUNLOG.md`, from journal `error` / `recovery` | done |
| **Manual-intervention count** | `RUNLOG.md` header, counted from `interventions.md` table rows | done |
| The raw log itself | `runs/<id>/journal.jsonl` | **not in the repo — see below** |

## 4 · Final submission and results summary

| Required | Where it is | Status |
|---|---|---|
| Final model output, starter-kit schema | `runs/<id>/final/submission.csv` — 170,588 rows, validated | produced; **not in the repo** |
| Results table: validation-best GAUC / nDCG@5 **+ absolute delta over baseline** | `RESULTS.md` via `orchestrator.report`; `01-results.md` | done |
| Resource usage: total tokens (in+out), agent wall-clock, iterations of 50 | `runs/<id>/summary.json` | done, per run |
| GPU-hours | **zero — every run was CPU-only** | n/a |
| Bonus: KuaiRand-1k / 27k | not attempted | n/a, does not reduce the Pure score |

## The gap that blocks 3, 3b, 4a and 4c

`runs/` is gitignored, so **none of these four are in the public repository**, and the
runs they refer to (`r20260831-0724`, `r20260831-0741`) were made on a Windows machine
and exist on no other. Deliverable 2 is a *public* repository; a path a reviewer cannot
open does not satisfy 3.

Whoever has those run directories fixes it in two commands:

```bash
python tools/archive_run.py runs/r20260831-0724     # carries deliverable 4a
python tools/archive_run.py runs/r20260831-0741     # the best clean score
git add runs/examples && git commit
```

`runs/examples/` is exempted from the ignore rules, including the `*.csv` rule that
would otherwise swallow `final/submission.csv` — which *is* deliverable 4a. The script
skips the per-node submissions that would make the commit hundreds of megabytes. See
`runs/examples/README.md`.

This also unblocks the `TODO`s in `docs/devpost.md`: the agent-hypothesis quotes can
only come from `journal.jsonl`, so nobody without those directories can fill them in.

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

1. **Archive the cited runs into `runs/examples/` and commit them** (see above). Until
   that is done, deliverables 3, 3b, 4a and 4c are not in the repository at all, and the
   devpost quotes cannot be filled in.
2. Read `docs/handover/01-results.md` and use only numbers marked trustworthy. **0.6189 and
   0.8484 are both leaked and must not appear as results.**
3. Fill the final numbers into `docs/devpost.md` from
   `runs/examples/r20260831-0741/summary.json`.
4. The agent-hypothesis quotes in `docs/devpost.md` are marked `TODO`; pull them verbatim from
   the winning run's journal (`event: "proposal"`, field `hypothesis`). Do not paraphrase —
   Innovation is scored on what the agent chose to try and why, and its own words are the
   evidence.
