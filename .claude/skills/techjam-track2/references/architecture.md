# Architecture and design rationale

## What we are building

An **autonomous ML research agent** for KuaiRand-Pure. An orchestrator drives a closed loop:
an LLM writes a complete `pipeline.py`, a sandbox runs it, an evaluator scores its submission,
a search policy decides what to try next, and a journal records the whole trajectory.

The shape is deliberately close to AIDE (arXiv:2502.13138): **treat ML engineering as code
optimisation, and search over a tree of solution programs.** That framing is proven, it is
easy to explain to judges, and it maps cleanly onto four parallel workstreams.

## Data flow

```
                          +---------------------------+
                          |  TaskSpec + data card (C) |
                          +-------------+-------------+
                                        |
   +-------------+   context   +--------v---------+   Proposal   +----------------+
   | Knowledge   +------------>+   Agent runtime  +------------->+   Solution     |
   | base (C)    |   top-K     |   (B) LLM calls  |  full file   |   node (A)     |
   +-------------+   ideas     +------------------+              +--------+-------+
                                        ^                                 |
                                        | repair on error                 | pipeline.py
                                        |                                 v
   +-------------+             +--------+---------+              +----------------+
   | Journal (D) +<------------+  Orchestrator    +<-------------+  Sandbox exec  |
   | jsonl       |   every     |  loop + search   |  ExecResult  |  (B)           |
   +------+------+   event     |  (A)             |              +--------+-------+
          |                    +--------+---------+                       |
          v                             ^                                 v
   +-------------+                      |            metrics     +----------------+
   | Dashboard,  |                      +------------------------+  Evaluator (C) |
   | RESULTS.md  |                                               +----------------+
   | (D)         |
   +-------------+
```

## The loop, precisely (Role A implements this)

```
for iteration in 1..50, while wall_clock < 6h and not converged:
    kind, parent = policy.next_action(tree)

    if kind == "draft":    proposal = agent.draft(ctx)
    if kind == "improve":  proposal = agent.improve(ctx, parent)
    if kind == "debug":    proposal = agent.repair(ctx, parent)

    node = tree.add(parent, kind, proposal)
    write proposal.code -> node.workspace/pipeline.py
    result = executor.run(node, split="val", seed=0, timeout_s=...)

    if not result.ok:
        journal(error); node.status = classify(result)
        if node.repair_attempts < 3: schedule a debug node next iteration
        else: node.status = "dead"; journal(recovery, "route_around")
        continue

    ok, msg = evaluator.validate(result.artifacts["submission"], "val")
    if not ok: treat as error_class="contract" and repair

    node.metrics = evaluator.score(result.artifacts["submission"], "val")
    journal(eval); update best; check convergence
```

Notes that matter:

- **A failed node never stops the run.** Three repair attempts, then mark dead and route
  around by selecting a different parent. That behaviour *is* the Robustness score.
- **Convergence is on validation primary**: no improvement > 0.002 over 3 consecutive
  *scored* iterations. Errored iterations do not count toward N.
- **The final submission is the validation-best node, retrained with `--split test`**
  (train + validation), then validated with `submit.py --check`. Do this once, at the end.
- **Everything is checkpointed.** `state.json` is rewritten atomically every iteration so a
  crashed run resumes without a human, which protects the intervention count.

## Search policy (Role A)

Start simple and only add complexity if the data says to:

1. **Draft phase** (iterations 1–3): three independent drafts from different angles
   (reproduce-the-baseline, feature-first, model-first). Cheap insurance against a bad start.
2. **Greedy improve**: usually expand the current validation-best node.
3. **Debug-first**: if any node is `buggy` with `repair_attempts < 3`, fix it before improving.
   A broken node is worth more than a new idea — it is already halfway there.
4. **Explore**: with probability ~0.2, or after 3 consecutive non-improving iterations,
   expand the second-best distinct node instead. This is what stops the run from stalling.
5. **Anti-overfit guard**: hidden test is what counts. Prefer a node whose gain over its parent
   exceeds 2 sigma (0.0016) of baseline seed noise; log the gain so D can plot it. If the top
   two nodes are within noise, prefer the simpler one (fewer lines changed).

## Why these boundaries

| Boundary | Why it is a good seam |
|---|---|
| `pipeline.py` as a single file with a CLI | The LLM's entire output surface. Sandboxing, diffing and reverting all become trivial. It also means Role B and Role C never need to touch each other's code. |
| Scoring outside the pipeline | Pins the metric conventions, stops the agent from accidentally (or conveniently) scoring itself wrong, and lets C change the evaluator without touching the agent. |
| `Proposal` carries `hypothesis` | Forces the LLM to state *why* before *what*, which is literally the Innovation & Problem Insight rubric. Also makes the journal readable. |
| Journal as append-only JSONL | Decouples D entirely: the dashboard is a pure function of the journal, so D can build against a fixture file before the orchestrator exists. |
| Knowledge base behind `retrieve()` | Lets C keep adding ideas all weekend without ever touching orchestration code. |

## Non-goals (say no to these)

- A web UI. Nobody scores it. A static HTML report is enough and D already owns it.
- Multi-agent role-play inside the agent runtime (planner/coder/critic personas). It burns
  tokens, which are scored, and adds failure modes. One well-prompted call per node.
- A general-purpose framework for arbitrary datasets. KuaiRand-Pure is 100% of the score.
  Generality only earns points if it costs nothing.
- Distributed or GPU training. The reference pipeline needs ~28 min of one CPU core for 100
  iterations. Wall-clock is scored; a GPU will not help and may hurt.
- Chasing the bonus benchmarks before the required one converges and beats the baseline.

## Risk register

| Risk | Mitigation | Owner |
|---|---|---|
| Agent overfits validation, loses on hidden test | Seed-averaged validation on the final candidate; prefer simpler node within noise; never select on a single seed | A + C |
| Run dies at hour 4 and nobody notices | Atomic checkpoint + `--resume`; D's dashboard regenerates from journal at any time | A + D |
| LLM keeps producing the same idea | `tried` idea-ids passed into `retrieve()`; dedupe proposals by normalised code hash; force explore after 3 flat iterations | A + C |
| Token cost blows the Feasibility tier | Hard token budget in `Context`; never send full data or full history; prompt-cache the static task card | B |
| Generated code imports something not installed | `requirements-pipeline.txt` is the whitelist, stated in the prompt; `import` error class triggers a repair that must fall back | B + C |
| We beat the baseline but the submission CSV is malformed | `submit.py --check` runs on every scored node, not just the final one | C |
| Nobody has time to write the Devpost entry | D starts it at H+12 and keeps it current, not at H+60 | D |
