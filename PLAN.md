# TechJam 2026 Track 2 — Team Plan

**Track:** Autonomous ML Research Agent for Recommender Systems (KuaiRand-Pure)
**Window:** 29 Aug 12:00 SGT → 1 Sep 12:00 SGT (72 h)
**Team:** 4 people, each driving Claude Code

## What we are building

An **autonomous ML research agent**. Given the KuaiRand-Pure dataset and a metric, an LLM-driven
orchestrator writes its own pipeline code, trains it, scores it, reflects on the result, revises,
and repeats — up to 50 iterations or 6 hours — with as close to zero human intervention as we can
manage. Architecturally it is AIDE-style (arXiv:2502.13138): treat ML engineering as code
optimisation and search over a tree of solution programs.

The number to beat: official baseline **validation primary 0.6016**, hidden-test **0.5946**.
The attainable ceiling is **0.8645**, not 1.0 — 27% of users have no positive label.

## How we are scored, and who owns each score

| Criterion | Owner |
|---|---|
| Primary metric (hidden-test delta over baseline) | C (ideas) + A (search) |
| Robustness (how failures are handled) | B (repair loop) + A (routing) |
| Innovation & Problem Insight (what it tried, and why) | C (cited idea bank) + B (hypothesis prompts) |
| Autonomy (number of manual interventions) | everyone |
| Feasibility (tokens + wall-clock) | B (prompts) + D (accounting) |
| Presentation | D |

## The four roles

| | Role | Owns | The thing that must not fail |
|---|---|---|---|
| **A** | Orchestrator & Search | loop, solution tree, convergence, budget, checkpoint/resume, search policy | a 50-iteration run finishing unattended |
| **B** | Agent Runtime & Sandbox | Claude API client, prompts, code emission, sandboxed execution, error classification, repair loop | every injected fault recovers with no human |
| **C** | ML Core & Knowledge | baseline reproduction, evaluator, data card, idea bank, feature cache | the baseline reproduces and the metrics are exact |
| **D** | Telemetry & Reporting | journal, accounting, dashboard, RESULTS.md, packaging, Devpost, video | a valid submission exists from H+60 onward |

Full scope, acceptance tests and traps per role are in
`.claude/skills/techjam-track2/references/roles/`.

## How the four of you work in parallel

Each person opens Claude Code in this repo and starts every session with:

```
/techjam-track2 A
```

(substituting their role letter). The skill loads the problem digest, the frozen contracts,
their own scope, and the current checkpoint. Nobody re-plans; everybody plans their next task.

Three rules make the parallelism work:

1. **Frozen contracts by H+2.** A writes `orchestrator/contracts.py` and the stubs in the first
   two hours. Everyone else codes against those shapes, stubbing whatever does not exist yet.
2. **Strict directory ownership.** You edit only what your role file lists. Cross-team requests
   go in `STATUS.md`, not into someone else's module.
3. **`make check` stays green.** Lint + unit tests + a 3-iteration smoke run with a stubbed LLM,
   under 60 seconds. Red `main` is the team's top priority.

## Schedule at a glance

| Checkpoint | What must be true |
|---|---|
| **H+2** | Contracts frozen |
| **H+6** | Baseline reproduced (0.6016); everyone's piece runs against stubs |
| **H+12** | First agent-written `pipeline.py` runs and gets scored |
| **H+24** | 8-iteration unattended run produces a scored submission. **Tag it.** |
| **H+36** | Validation primary > 0.6016 from a fully autonomous run |
| **H+42** | First full 50-iteration / 6 h official run completes |
| **H+48** | Official run #2 launched — nobody touches it |
| **H+60** | **Complete valid submission on Devpost.** Non-negotiable. |
| **H+70** | Code freeze. Buffer only after this. |
| **H+72** | Deadline. No late submissions. |

Detail, including the sleep rota and the scope-cut order, is in
`.claude/skills/techjam-track2/references/timeline.md`.

## Setup checklist before H+0

- [ ] All four registered on **both** the Registration Form and Devpost (both are required)
- [ ] GitHub repo created, all four with push access, this directory pushed
- [ ] `kuairand-starter-kit.zip` downloaded from the Lark doc (Section 2.4)
- [ ] KuaiRand-Pure data downloaded from https://kuairand.com
- [ ] `ANTHROPIC_API_KEY` in each person's environment — never committed
- [ ] Python 3.11 + venv on the machine that will host the official runs
- [ ] Group chat for the H+6 / H+12 / H+24 / H+36 / H+42 / H+48 / H+60 syncs

## The two rules that can lose us everything

1. **No external training data.** KuaiRand only. This is the single disqualifying rule.
2. **Nothing after H+60 may leave us without a submittable project.** From H+60 we always have
   a complete, valid Devpost entry; everything later is an improvement on top of it.
