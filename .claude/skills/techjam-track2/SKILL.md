---
name: techjam-track2
description: Team coordination skill for TikTok TechJam 2026 Track 2 (Autonomous ML Research Agent for Recommender Systems). Invoke at the start of every working session with your role letter (A, B, C, or D) to load your workstream's scope, the frozen cross-team contracts, acceptance tests, and the 72-hour schedule. Use whenever working in this repo on the orchestrator, agent runtime, ML core, or reporting.
---

# TechJam 2026 — Track 2 Team Skill

You are one of four Claude Code agents building a single project. This skill tells you
**what you own, what you must not touch, and the interfaces you must honour** so four
people can work in parallel without blocking or colliding.

## How to use

```
/techjam-track2 A     # Orchestrator & Search
/techjam-track2 B     # Agent Runtime & Execution Sandbox
/techjam-track2 C     # ML Core, Data & Knowledge
/techjam-track2 D     # Telemetry, Reporting & Submission
```

If no role is given, ask which role, then continue. Never guess.

**On invocation, in this order:**
1. Read `references/problem.md` — what we are being scored on. Non-negotiable.
2. Read `references/contracts.md` — the frozen interfaces. Never change these unilaterally.
3. Read `references/roles/<your-role>.md` — your scope and acceptance tests.
4. Read `references/timeline.md` — find the current checkpoint and work toward it.
5. Read `STATUS.md` at repo root (if present) for what the other three have landed.

Then get to work. Do not re-plan the project; the plan exists. Plan *your* next task only.

## The one-paragraph brief

Build an **autonomous ML research agent**: an LLM-driven orchestrator that, given the
KuaiRand-Pure dataset and a metric, writes its own pipeline code, trains, evaluates,
reflects on the result, revises, and repeats — for up to 50 iterations or 6 hours — with
as close to zero human intervention as possible. Ranking is on a hidden test set. We are
scored on: score delta over the official baseline, robustness under failure, the quality
of the ideas the agent chose to try, how few humans it needed, and how cheap the run was.

## Rules that apply to everyone

1. **Stay in your lane.** Only edit files under the paths your role file lists as *owned*.
   If you need a change in someone else's module, write the request into `STATUS.md`
   under `## Requests` and code against the existing contract in the meantime.
2. **Contracts are frozen after H+2.** Everything in `references/contracts.md` is a hard
   interface. If you genuinely must change one, edit `STATUS.md` under `## Contract change
   proposed`, and keep the old shape working until the team confirms.
3. **Stub, never block.** If your work depends on a module that does not exist yet, write
   a minimal stub behind the contract's interface in `tests/stubs/` and keep going.
4. **Every merge keeps `make check` green.** Lint + unit tests + a 3-iteration smoke run
   with a stubbed LLM. If you break it, fix it before anything else.
5. **The journal is a first-class deliverable, not a log file.** Judges read it to score
   Autonomy, Robustness and Innovation. Every hypothesis, diff, metric, error and recovery
   goes in, in the schema in `contracts.md`. Never print-debug into it.
6. **Log human touches.** During any official run, every manual intervention gets a line in
   `runs/<id>/interventions.md`. This number is directly scored — fewer is better. Treat a
   manual fix as a bug in the agent, and fix the agent instead wherever you can.
7. **Token discipline.** LLM spend is scored. No speculative fan-out, no re-sending the
   whole dataset, no 200k-token prompts. Cache aggressively.
8. **Never touch the hidden test set for development.** Train + validation only.
9. **No external training data.** Only KuaiRand. This is the one hard disqualifying rule.
10. **Commit small, push often, rebase on `main`.** Branch is `feat/<role-letter>-<topic>`.

## Repo layout and ownership

```
orchestrator/
  core/        A   loop, solution tree, convergence, budget, checkpoint/resume
  search/      A   node selection policy (draft / improve / debug)
  agent/       B   LLM client, prompt assembly, proposal parsing, repair loop
  exec/        B   sandboxed subprocess runner, timeouts, error classification
  eval/        C   evaluate.py wrapper, submission validation, data cards
  knowledge/   C   RecSys idea bank the agent draws hypotheses from
  report/      D   journal -> dashboard, results table, resource accounting
  contracts.py A   the frozen dataclasses (change = team decision)
data/          C   splits, cached feature matrices (gitignored)
runs/          -   per-run workspaces + journal.jsonl (gitignored except examples)
docs/          D   README, architecture diagram, Devpost writeup, RESULTS.md
tests/         all everyone tests their own module; D owns the smoke run
```

## Working style for this repo

- Prefer **one self-contained `pipeline.py` per solution node** over a plugin framework.
  The LLM writes that whole file; we diff it after. Simple to sandbox, simple to revert.
- Prefer **deterministic, seeded, cached** ML code. Reruns must be reproducible.
- Prefer **failing loudly inside the sandbox and recovering in the orchestrator** over
  defensive try/except scattered through generated code.
- When in doubt about scope, do the smaller thing that makes the run work end-to-end.
  An end-to-end mediocre agent beats a half-built brilliant one at H+72.

## Reference files

- `references/problem.md` — problem statement digest, scoring, constraints, key numbers
- `references/contracts.md` — frozen interfaces, journal schema, pipeline CLI contract
- `references/architecture.md` — module map, data flow, design rationale
- `references/timeline.md` — 72-hour schedule with hard checkpoints
- `references/roles/a-orchestrator.md`
- `references/roles/b-agent-runtime.md`
- `references/roles/c-ml-core.md`
- `references/roles/d-telemetry-report.md`
