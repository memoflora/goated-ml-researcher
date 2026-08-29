# goated-ml-researcher

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**

An LLM-driven orchestrator that runs the machine-learning engineering loop by itself on
KuaiRand-Pure: read the problem, inspect the data, engineer features, train and tune, evaluate,
reflect, revise — and repeat, up to 50 iterations or 6 hours, with as close to zero human
intervention as we can manage. It writes its own pipeline code every round.

Architecturally this is AIDE-shaped ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)):
treat ML engineering as code optimisation and search over a tree of solution programs.

## The number to beat

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline — validation | 0.6674 | 0.5357 | **0.6016** |
| Official baseline — hidden test | 0.6610 | 0.5282 | **0.5946** |
| Perfect ranking (attainable ceiling) | 1.0000 | 0.7289 | **0.8645** |

The ceiling is 0.8645, not 1.0 — 27.1% of hidden-test users have no positive label, so their
nDCG is 0 for any model. Random scoring sits at 0.4753. Judge progress against 0.8645.

## Start here

| Document | What it is |
|---|---|
| [PLAN.md](PLAN.md) | The 72-hour plan: roles, checkpoints, setup checklist |
| [STATUS.md](STATUS.md) | Shared scratchpad — update your section before every sync |
| [`references/problem.md`](.claude/skills/techjam-track2/references/problem.md) | Problem statement digest, scoring rubric, constraints |
| [`references/contracts.md`](.claude/skills/techjam-track2/references/contracts.md) | The frozen interfaces between the four workstreams |
| [`references/architecture.md`](.claude/skills/techjam-track2/references/architecture.md) | Module map, data flow, design rationale, risk register |
| [`references/timeline.md`](.claude/skills/techjam-track2/references/timeline.md) | Hour-by-hour schedule and scope-cut order |

## How the team works

Four people, each driving Claude Code. Every working session starts by loading the team skill
with your role letter:

```bash
/techjam-track2 A
```

That loads the problem digest, the frozen contracts, your own scope with acceptance tests and
traps, and the current checkpoint. Nobody re-plans the project; everybody plans their next task.
The skill lives in [`.claude/skills/techjam-track2/`](.claude/skills/techjam-track2/) and is
picked up automatically when Claude Code is opened in this repo.

| | Role | Owns | Must not fail |
|---|---|---|---|
| **A** | Orchestrator & Search | loop, solution tree, convergence, budget, checkpoint/resume, search policy | a 50-iteration run finishing unattended |
| **B** | Agent Runtime & Sandbox | Claude API client, prompts, code emission, sandboxed execution, error classification, repair loop | every injected fault recovers with no human |
| **C** | ML Core & Knowledge | baseline reproduction, evaluator, data card, cited idea bank, feature cache | the baseline reproduces and the metrics are exact |
| **D** | Telemetry & Reporting | journal, accounting, dashboard, RESULTS.md, packaging, Devpost, video | a valid submission exists from H+60 onward |

Three rules make four parallel agents compose:

1. **Contracts frozen at H+2.** A ships `orchestrator/contracts.py` plus stubs first; everyone
   else codes against those shapes and stubs whatever does not exist yet.
2. **Strict directory ownership.** You edit only what your role file lists. Cross-team requests
   go in `STATUS.md`, never into someone else's module.
3. **`make check` stays green.** Lint + unit tests + a 3-iteration smoke run with a stubbed LLM,
   under 60 seconds.

## Planned layout

```
orchestrator/
  core/        A   loop, solution tree, convergence, budget, checkpoint/resume
  search/      A   node selection policy (draft / improve / debug)
  agent/       B   LLM client, prompt assembly, proposal parsing, repair loop
  exec/        B   sandboxed subprocess runner, timeouts, error classification
  eval/        C   evaluate.py wrapper, submission validation, data cards
  knowledge/   C   RecSys idea bank the agent draws hypotheses from
  report/      D   journal -> dashboard, results table, resource accounting
  contracts.py A   the frozen dataclasses
data/          C   splits, cached feature matrices (gitignored)
runs/          -   per-run workspaces + journal.jsonl (gitignored except examples)
docs/          D   README, architecture diagram, Devpost writeup, RESULTS.md
tests/         all everyone tests their own module; D owns the smoke run
```

## Setup

- [ ] All four registered on **both** the Registration Form and Devpost — both are required
- [ ] `kuairand-starter-kit.zip` downloaded from the Lark information document, Section 2.4
- [ ] KuaiRand-Pure data from [kuairand.com](https://kuairand.com)
- [ ] `ANTHROPIC_API_KEY` in your environment — never committed
- [ ] Python 3.11 + venv on the machine that will host the official runs

## The two rules that can lose us everything

1. **No external training data.** KuaiRand only — no augmenting, joining, or pre-training on any
   other dataset, and no pretrained weights trained on these benchmarks' test labels. This is
   the single disqualifying rule.
2. **Nothing after H+60 may leave us without a submittable project.** From H+60 there is always
   a complete, valid Devpost entry; everything later is an improvement on top of it.
