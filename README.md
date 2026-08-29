# goated-ml-researcher

**TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems**

An LLM-driven orchestrator that runs the ML engineering loop by itself on KuaiRand-Pure: read
the problem, inspect the data, engineer features, train and tune, evaluate, reflect, revise —
repeating up to 50 iterations or 6 hours with as close to zero human intervention as we can
manage. It writes its own pipeline code every round.

AIDE-shaped ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)): ML engineering as code
optimisation, searched over a tree of solution programs.

## The number to beat

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline — validation | 0.6674 | 0.5357 | **0.6016** |
| Official baseline — hidden test | 0.6610 | 0.5282 | **0.5946** |
| Perfect ranking (attainable ceiling) | 1.0000 | 0.7289 | **0.8645** |

The ceiling is 0.8645, not 1.0 — 27.1% of hidden-test users have no positive label, so their
nDCG is 0 for any model. Random scoring sits at 0.4753.

## The team

Two ML engineers build the agent; two ML researchers decide what it knows and what it tries.
Each person owns exactly one judged criterion.

| | Role | Owns | Owns the score for |
|---|---|---|---|
| **A** | ML Engineer — Orchestrator & Run | `contracts.py` `core.py` `policy.py` `journal.py` `run.py` | Autonomy |
| **B** | ML Engineer — Agent Runtime & Sandbox | `agent.py` `sandbox.py` `Makefile` `.github/` | Robustness + Feasibility |
| **C** | ML Researcher — Data, Metrics & Evaluation | `evaluate.py` `datacard.py` `report.py` `data/` | Primary metric |
| **D** | ML Researcher — Method, Knowledge & Story | `ideas.yaml` `knowledge.py` `prompts/` `reference/` `docs/` | Innovation + Presentation |

Ownership is per file, not per directory. B owns the prompt plumbing; D owns the prompt text.

Open Claude Code in this repo and start every session with your letter:

```bash
/techjam-track2 A
```

That loads the problem digest, the frozen contracts, your section of the role file, and the
current checkpoint. The skill lives in [`.claude/skills/techjam-track2/`](.claude/skills/techjam-track2)
and is picked up automatically on clone.

## Documents

| | |
|---|---|
| [PLAN.md](PLAN.md) | Roles, the 72-hour schedule, standing rules, setup checklist |
| [STATUS.md](STATUS.md) | Shared scratchpad — update before every sync |
| [problem.md](.claude/skills/techjam-track2/references/problem.md) | Problem statement digest, scoring rubric, constraints |
| [contracts.md](.claude/skills/techjam-track2/references/contracts.md) | Architecture, module layout, frozen interfaces, journal schema |
| [roles.md](.claude/skills/techjam-track2/references/roles.md) | All four roles: build order, acceptance tests, traps |

## Planned layout

One file per concern — no nested packages.

```
orchestrator/
  contracts.py    A   frozen dataclasses
  core.py         A   loop, tree, convergence, budget, resume
  policy.py       A   draft / improve / debug / explore selection
  journal.py      A   append-only JSONL + token & wall-clock accounting
  run.py          A   CLI
  agent.py        B   LLM client, prompt assembly, proposal parsing, repair
  sandbox.py      B   subprocess runner, timeouts, error classification
  evaluate.py     C   score() and validate()
  datacard.py     C   the EDA summary the LLM reads
  report.py       C   journal -> RESULTS.md + trajectory PNG
  knowledge.py    D   retrieve()
  ideas.yaml      D   the idea bank
  prompts/        D   system.md, draft.md, improve.md, repair.md
data/             C   splits and caches (gitignored)
reference/        D   hand-written calibration pipeline
runs/             -   per-run workspaces + journal.jsonl (gitignored)
docs/             D   Devpost writeup, diagram
tests/            all B owns the smoke run
```

## Setup

- [ ] Registered on **both** the Registration Form and Devpost — both are required
- [ ] `kuairand-starter-kit.zip` from the Lark information document, §2.4
- [ ] KuaiRand-Pure data from [kuairand.com](https://kuairand.com)
- [ ] `ANTHROPIC_API_KEY` in your environment — never committed
- [ ] Python 3.11 + venv

## The two rules that can lose us everything

1. **No external training data.** KuaiRand only — no augmenting, joining, or pre-training on any
   other dataset, and no pretrained weights trained on these benchmarks' test labels. The single
   disqualifying rule.
2. **Nothing after H+60 leaves us without a submittable project.**
