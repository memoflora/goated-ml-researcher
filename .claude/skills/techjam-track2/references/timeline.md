# 72-hour schedule

Window: **29 Aug 12:00 SGT → 1 Sep 12:00 SGT.** All hours below are hours-since-start (H+n).
H+0 = 29 Aug 12:00. H+72 = 1 Sep 12:00 = hard deadline, no late submissions.

Sync points are 15 minutes, camera optional, everyone posts to `STATUS.md` first.

---

## Phase 0 — Foundations (H+0 → H+6)

Goal: **everyone can run something end to end, and the contracts are frozen.**

| Who | Deliverable by H+6 |
|---|---|
| A | `orchestrator/contracts.py` complete and frozen by **H+2**. Loop skeleton with stub agent + stub executor runs 3 iterations and writes a well-formed journal. |
| B | Sandbox executor runs an arbitrary `pipeline.py` with timeout + capture. Claude API call returns a parsed `Proposal` for a toy prompt. |
| C | Starter kit unpacked, data downloaded, **official baseline reproduced** (val primary ≈ 0.6016). `Evaluator.score()` matches published numbers. |
| D | Repo, CI, `make check`, journal fixture file, dashboard renders the fixture. |

**H+2 checkpoint (hard):** contracts frozen. If they are not, everything after this slips.
**H+6 sync:** everyone demos their piece against a stub. Any blocked dependency becomes a stub.

---

## Phase 1 — First autonomous run (H+6 → H+24)

Goal: **an unattended 5-iteration run that writes real pipelines, scores them, and logs it all.**

| Who | Deliverable by H+24 |
|---|---|
| A | Real loop: tree, greedy improve, debug-first, convergence detector, budget guard, `--resume`. |
| B | Real drafting + improving prompts, full-file code emission, repair loop with error classification, token accounting emitted. |
| C | Data card (EDA summary the LLM reads), idea bank tiers T0–T2 populated, submission validator wired in, subsample mode for fast iteration. |
| D | Dashboard reads real journals: score trajectory, tree view, hypothesis table, resource totals. Devpost draft started. |

**H+12 checkpoint:** first agent-written `pipeline.py` that runs and scores, even if worse
than baseline. This is the riskiest moment of the weekend — if it slips, cut scope, not this.

**H+24 checkpoint (hard):** `dev` mode run completes 8 iterations unattended, produces a
scored submission, dashboard renders it. **Tag this commit.** From here we always have
something submittable.

---

## Phase 2 — Make it good (H+24 → H+42)

Goal: **beat the baseline on validation, and survive 50 iterations without a human.**

| Who | Focus |
|---|---|
| A | Explore/exploit policy, anti-stall rules, dead-node routing, anti-overfit selection, seed-averaged final scoring. |
| B | Prompt-cache the static task card, trim context, robustness hardening: inject faults (syntax error, infinite loop, OOM, missing file, NaN scores, wrong CSV schema) and prove each recovers with zero human input. |
| C | Idea bank T3–T4 with citations, feature cache, faster training loop, hand-written reference pipeline that beats baseline (proof the headroom exists so we know when the *agent* is the bottleneck). |
| D | Resource accounting exact, RESULTS.md generator, architecture diagram, README reproduction steps, Devpost draft at 80%. |

**H+36 checkpoint:** validation primary > 0.6016 from a fully autonomous run.
**H+42 checkpoint (hard):** first full `official` run (50 iterations / 6 h) launched and
finishing without intervention. Whatever it scores, we now know our real cost and duration.

---

## Phase 3 — The scored run (H+42 → H+60)

Goal: **the run we actually submit.**

- H+42 → H+48: read the first official run's journal together. Fix the top three failure
  modes. Do **not** add features; only remove reasons the agent got stuck.
- H+48: launch **official run #2**. Nobody touches it. Every intervention is logged and costs us.
- In parallel: C attempts KuaiRand-1k as bonus *only if* Pure has already beaten the baseline.
- H+54: run #2 converges. Retrain the validation-best node with `--split test`, validate the
  CSV with `submit.py --check`, freeze `final/submission.csv`.

**H+60 checkpoint (hard, non-negotiable):** a complete, valid submission exists on Devpost —
code repo, run logs, submission CSV, results table, resource summary, description. It can be
improved after this, but from H+60 onward we are never in a state where we would submit nothing.

---

## Phase 4 — Land it (H+60 → H+72)

- H+60 → H+66: demo video (3 min: the loop running, the journal, the trajectory plot, the
  numbers). Devpost description final. README reproduction check on a clean clone.
- H+66 → H+70: optional third official run only if it is strictly better and finishes by H+70.
  Bonus benchmark results included only if validated.
- H+70 → H+72: buffer. Final Devpost edit. Do not start anything new.

**Do not push code after H+70.** Deadlines slip; ours will not.

---

## Standing rules

- **Sync at H+6, H+12, H+24, H+36, H+42, H+48, H+60.** 15 minutes, `STATUS.md` first.
- **Sleep is scheduled, not optional.** Suggested: A+C sleep H+14→H+21, B+D sleep H+21→H+28,
  so someone is always awake to watch a run. A tired team writes the bugs that cost the
  intervention count.
- **Never leave `main` red.** If `make check` fails, that is the whole team's top priority.
- **The intervention counter is sacred.** During official runs, if you want to touch the
  machine, ask in the group chat first and log it in `runs/<id>/interventions.md`.
- **Cut scope in this order** when behind: bonus benchmarks → tree search sophistication →
  dashboard polish → idea bank depth. Never cut: the end-to-end loop, the journal, the
  submission validator, the resource accounting.
