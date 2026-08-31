---
name: techjam-track2
description: Team coordination skill for TikTok TechJam 2026 Track 2 (Autonomous ML Research Agent for Recommender Systems). Invoke at the start of every working session with your role letter (A, B, C, or D) to load your scope, the frozen cross-team contracts, acceptance tests, and the current checkpoint. Use whenever working in this repo on the orchestrator, agent runtime, evaluation, or the idea bank.
---

# TechJam 2026 — Track 2 Team Skill

Four people, each driving Claude Code, building one project. Two ML engineers build the agent;
two ML researchers decide what it knows and what it tries. This skill tells you **what you own,
what you must not touch, and the interfaces you must honour.**

## How to use

```
/techjam-track2 A     # ML Engineer   — Orchestrator & Run
/techjam-track2 B     # ML Engineer   — Agent Runtime & Sandbox
/techjam-track2 C     # ML Researcher — Data, Metrics & Evaluation
/techjam-track2 D     # ML Researcher — Method, Knowledge & Story
```

No role given? Ask which one. Never guess.

## Before you report any number

**`docs/handover/` is the source of truth for results.** Two of our headline numbers turned
out to be label leakage and were withdrawn; `docs/handover/01-results.md` marks which figures
survived the audit. Never quote a score from this project without checking there first —
0.6189 and 0.8484 are both invalid and must not appear in the writeup.

| | |
|---|---|
| `docs/handover/README.md` | start here; the one-paragraph version and what to lead with |
| `docs/handover/01-results.md` | every number, its provenance, and whether it is trustworthy |
| `docs/handover/02-what-we-built.md` | the system and the design decisions |
| `docs/handover/03-findings.md` | what we discovered, ordered by what matters |
| `docs/handover/04-run-history.md` | every live run and what it taught |
| `docs/handover/05-deliverables.md` | the organisers' required deliverables, mapped |

**On invocation, read in this order:**

1. `references/problem.md` — what we are scored on. Non-negotiable.
2. `references/starter-kit-findings.md` — measured dead ends and the organisers' ranked list of
   where the headroom is. **Read before writing any idea, prompt, or pipeline.**
3. `references/contracts.md` — architecture and the frozen interfaces.
4. `references/roles.md` — find **your section**; skim the other three so you know the seams.
5. `docs/handover/` — the results, the findings, and the run history. Read this before
   writing anything that quotes a number.
6. `STATUS.md` and `PLAN.md` at repo root — what has landed, and which checkpoint is next.

Then work. The project is already planned; plan *your next task* only.

## The brief

Build an autonomous ML research agent: an LLM-driven orchestrator that, given a dataset, a
problem statement and a metric, writes its own pipeline code, trains, evaluates, reflects,
revises, and repeats — up to 50 iterations or 6 hours, with as close to zero human intervention
as possible. Ranking is on a hidden test set.

**The task is configuration.** A problem is a file in `tasks/<name>.yaml` naming the data,
target, split, metrics and submission schema; the data card, evaluator, prompts and report all
read their facts from there. KuaiRand-Pure is one such file — and still 100% of the score. We are scored on: score delta over the official baseline, robustness under
failure, the quality of the ideas the agent chose to try, how few humans it needed, and how
cheap the run was.

The baseline to beat is **validation primary 0.6016**. The attainable ceiling is **0.8645**,
not 1.0 — 27% of users have no positive label.

## Who owns what

| | Role | Owns files | Owns the score for |
|---|---|---|---|
| **A** | ML Engineer — Orchestrator & Run | `contracts.py` `core.py` `policy.py` `journal.py` `run.py` | Autonomy |
| **B** | ML Engineer — Agent Runtime & Sandbox | `agent.py` `sandbox.py` `Makefile` `.github/` | Robustness + Feasibility |
| **C** | ML Researcher — Data, Metrics & Evaluation | `evaluate.py` `datacard.py` `report.py` `splits.py` `taskspec.py` `metrics.py` `datasource.py` `profile.py` `data/` `tasks/` | Primary metric |
| **D** | ML Researcher — Method, Knowledge & Story | `ideas.yaml` `knowledge.py` `prompts/` `reference/` `docs/` | Innovation + Presentation |

Ownership is **per file**, not per directory. B owns prompt plumbing; **D owns prompt text**.

## Rules that apply to everyone

1. **Stay in your lane.** Only edit the files your section lists. Need a change elsewhere? Put
   it in `STATUS.md` under `## Requests` and code against the existing contract meanwhile.
2. **Contracts freeze at H+2.** To change one, add a `## Contract change proposed` entry in
   `STATUS.md`, keep the old shape working, get one ack.
3. **Stub, never block.** Missing dependency? Write a minimal stub behind its interface in
   `tests/stubs/` and keep moving.
4. **`make check` stays green** — lint + tests + a 3-iteration smoke run with a stubbed LLM,
   under 60 s. Breaking it is the team's top priority.
5. **The journal is a deliverable, not a log file.** Judges read it to score Autonomy,
   Robustness and Innovation. Every hypothesis, metric, error and recovery goes in, in schema.
6. **Log every human touch** in `runs/<id>/interventions.md` during official runs. The count is
   directly scored. Treat a manual fix as a bug in the agent, and fix the agent instead.
7. **Token discipline.** LLM spend is scored. No fan-out, no full history, no 200k prompts.
8. **Train + validation only.** The environment now enforces this rather than asking: on any
   split the pipeline may not see, `masking.py` removes the label *and every other outcome of
   the impression* — `is_click`, `play_time_ms`, the likes. That exists because the agent
   leaked twice and scored the oracle ceiling; a prompt-level rule did not stop it. See
   `docs/handover/03-findings.md` §1.
9. **No external training data.** KuaiRand only. This is the one disqualifying rule.
10. **Small commits, push often, rebase on `main`.** Branch: `feat/<letter>-<topic>`.

## Working style

- One self-contained `pipeline.py` per solution node, behind a fixed CLI. Not a framework.
- Deterministic, seeded, cached ML code. Reruns must reproduce.
- Fail loudly inside the sandbox; recover in the orchestrator. Not defensive try/except
  scattered through generated code.
- In doubt about scope, do the smaller thing that keeps the run working end to end. An
  end-to-end mediocre agent beats a half-built brilliant one at H+72.

## Reference files

- `references/problem.md` — problem statement digest, scoring rubric, constraints, key numbers
- `references/starter-kit-findings.md` — verified baseline reproduction, measured dead ends, and
  the organisers' ranked list of unexplored directions
- `references/contracts.md` — architecture, module layout, frozen interfaces, journal schema
- `references/roles.md` — all four roles: scope, build order, acceptance tests, traps
- `tasks/kuairand-pure.yaml` — the scored task as configuration; copy it to add a new one
