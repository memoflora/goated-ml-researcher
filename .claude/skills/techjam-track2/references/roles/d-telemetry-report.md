# Role D — Telemetry, Reporting, Submission and Story

You own everything the judges actually look at. Three of the four required deliverables are
yours, and the resource numbers that gate the Feasibility score come out of your accounting.
This is not a support role — a great run with an unreadable log scores like a mediocre one.

## You own

```
orchestrator/report/journal.py     the Journal writer (everyone emits through it)
orchestrator/report/accounting.py  tokens, wall-clock, iterations, GPU-hours
orchestrator/report/dashboard.py   journal.jsonl -> static index.html
orchestrator/report/results.py     RESULTS.md generator
orchestrator/report/package.py     final submission packaging + checks
docs/                              README, architecture diagram, Devpost writeup
Makefile, .github/workflows/       CI
tests/test_journal.py, tests/test_dashboard.py, tests/fixtures/journal_sample.jsonl
```

## You do not touch

`orchestrator/core/`, `orchestrator/search/` (A) · `orchestrator/agent/`,
`orchestrator/exec/` (B) · `orchestrator/eval/`, `orchestrator/knowledge/` (C).

## Why you are not blocked on anyone

The dashboard is a pure function of `journal.jsonl`. Write `tests/fixtures/journal_sample.jsonl`
by hand in the first hour — a realistic 12-iteration run with two failures, one recovery, one
dead node and a plateau — and build everything against it. When A's real runs arrive, they
just work. **Do this before you write any other code.**

## Build order

1. **`Journal` writer** (due H+3, A is waiting on it). Append-only JSONL, one object per line,
   flush after every write, never buffer. Validate against the schema in
   `references/contracts.md` on write in debug mode. A malformed journal is a lost deliverable.
2. **Fixture + dashboard skeleton** (H+6). Single static `index.html`, no CDN, no external
   assets — it has to open from a file path on a judge's laptop.
3. **Accounting** (H+12). Cumulative `tokens_in` / `tokens_out`, wall-clock (agent total and
   per-node), iterations used out of 50, GPU-hours if any. Must be exact — this is a reported
   deliverable and it gates the Feasibility tier.
4. **Full dashboard** (H+24):
   - **score trajectory**: validation primary per iteration, with horizontal lines at the
     official baseline (0.6016 val) and the attainable ceiling (0.8645), plus a shaded band at
     ±0.0008 for seed noise. This single chart is the project's headline image.
   - **solution tree**: nodes coloured by status, edges parent→child, best node highlighted
   - **hypothesis table**: iteration, kind, hypothesis, idea ids, metric delta, outcome.
     Judges read this to score Innovation — make it skimmable.
   - **failure and recovery timeline**: every error, its class, and how it was resolved.
     This is the Robustness evidence.
   - **resource panel**: tokens, wall-clock, iterations, cost estimate
   - **intervention count**, displayed prominently. If it is zero, say zero, in large type.
5. **`RESULTS.md` generator** (H+36). Auto-generated, never hand-edited:

   | Benchmark | Metric | Official baseline | Ours (validation-best) | Absolute delta |
   |---|---|---|---|---|
   | KuaiRand-Pure | GAUC | 0.6674 | … | … |
   | KuaiRand-Pure | nDCG@5 | 0.5357 | … | … |
   | KuaiRand-Pure | primary | 0.6016 | … | … |

   Plus the resource block (tokens in/out, wall-clock, iterations used of 50, GPU-hours) and
   the intervention count. Report validation numbers and label them as validation — we never
   see hidden test.
6. **Submission packaging** (H+48). `make submit RUN=<id>` copies the final CSV, runs C's
   validator on it, regenerates RESULTS.md and the dashboard, and refuses to package if the
   submission fails validation.
7. **The story** (rolling from H+12, final H+66). Devpost description, README with a clean-clone
   reproduction path, a 3-minute demo video, and an architecture diagram. Draft early and keep
   it current — teams that start writing at H+60 submit worse projects than they built.

## CI and the Makefile

```
make check     lint + unit tests + smoke run (3 iterations, stubbed LLM), under 60 s
make dev       8-iteration run on subsampled data
make report RUN=<id>
make submit RUN=<id>
```

`make check` on every push. Keeping `main` green is your job to enforce, not just to run.

## The Devpost writeup — what to lead with

Judges scoring this track are looking for five things. Give them each a section:

1. **The loop, in one diagram.** Read problem → EDA → features → train/tune → evaluate →
   reflect, and where the LLM sits in it.
2. **The score.** Trajectory chart, absolute delta over the official baseline, framed against
   the 0.8645 attainable ceiling — not against 1.0. Explain the ceiling; most teams will not,
   and it shows we understood the metric.
3. **What the agent chose to try, and why.** Pull 4 or 5 of the best hypotheses verbatim from
   the journal, with the citation each drew on and the metric delta each produced. This is the
   Innovation section and quoting the agent is far more convincing than describing it.
4. **How it handled failure.** B's fault-injection table with real results, plus the recovery
   timeline from a real run.
5. **What it cost.** Tokens, wall-clock, iterations used, **and the intervention count.**

## Acceptance tests — you are done when

- [ ] `make report RUN=<id>` regenerates dashboard + RESULTS.md from `journal.jsonl` alone
- [ ] Dashboard opens from a `file://` path with no network, in both light and dark browsers
- [ ] Accounting totals match B's per-call token numbers exactly
- [ ] `make submit` refuses to package an invalid submission
- [ ] A clean clone plus README reproduces a `dev` run without asking anyone a question
- [ ] Devpost draft is complete at H+48 with placeholders only for the final numbers
- [ ] No API key, token, or secret appears anywhere in the repo, journal, or dashboard

## Traps

- **Never hand-edit RESULTS.md or the journal.** If a number is wrong, the generator is wrong.
  Judges can diff the journal against the table.
- **Do not let the dashboard depend on a CDN.** It has to work offline.
- **Do not report hidden-test numbers.** We never see them. Report validation and label it.
- Keep the raw journal in the repo for at least one full official run. It is the primary
  evidence for two of the four track criteria; a summary is not a substitute.
