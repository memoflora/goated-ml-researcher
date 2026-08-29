# The four roles

Two ML engineers build the agent. Two ML researchers decide what it knows, what it tries,
and how we prove it worked. Each of the four owns exactly one judged criterion.

| | Person | Owns files | Owns the score for |
|---|---|---|---|
| **A** | ML Engineer — Orchestrator & Run | `contracts.py` `core.py` `policy.py` `journal.py` `run.py` | **Autonomy** — runs finish unattended |
| **B** | ML Engineer — Agent Runtime & Sandbox | `agent.py` `sandbox.py` `Makefile` `.github/` | **Robustness + Feasibility** — failures recover, tokens stay cheap |
| **C** | ML Researcher — Data, Metrics & Evaluation | `evaluate.py` `datacard.py` `report.py` `data/` | **Primary metric** — the number is real and we can iterate fast |
| **D** | ML Researcher — Method, Knowledge & Story | `ideas.yaml` `knowledge.py` `prompts/` `reference/` `docs/` | **Innovation & Presentation** — what it tried, why, and how we tell it |

Ownership is per file, not per directory. If you need a change in someone else's file, put it
in `STATUS.md` under `## Requests` and code against the existing contract meanwhile.

The **A/B pair** builds the machine. The **C/D pair** decides what the machine thinks about.
They meet at two seams only: C's `datacard.py` and D's `prompts/` + `ideas.yaml` feed B's
`agent.py` through `Context`; C's `evaluate.py` feeds A's `core.py`. Both seams are frozen in
`contracts.md`, so all four can work at once from hour zero.

---

## A — ML Engineer: Orchestrator & Run

You own the loop. If a run stalls, crashes, or needs a human, that is your bug.

**Files:** `orchestrator/contracts.py`, `core.py`, `policy.py`, `journal.py`, `run.py`,
`tests/test_core.py`, `tests/test_policy.py`, `tests/stubs/`

### First task, due H+2 — everyone else is blocked on this

Write `contracts.py` exactly as specified in `contracts.md`, plus `tests/stubs/` with a
`StubAgent` (canned proposals from fixtures) and `StubExecutor` (canned results including
failures), and `journal.py` (append-only JSONL, flush every write). Push it, announce in
`STATUS.md`. Nothing else you do this weekend matters as much as shipping this in two hours.

### Then

1. **Loop** driving the stubs for 3 iterations, emitting journal events.
2. **Tree + best tracking.** `best` = highest validation primary among `status == "ok"` nodes.
3. **Convergence.** No improvement > `eps = 0.002` over the last `N = 3` **scored** iterations.
   Errored iterations do not count toward N. Also stop at 50 iterations or 6 h. Unit-test
   against synthetic sequences: plateau, noise, late jump.
4. **Budget guard.** Iterations, wall-clock, cumulative tokens. Refuse to start an iteration
   that cannot finish inside the ceiling.
5. **Checkpoint and resume.** Atomic `state.json` (temp → fsync → rename) every iteration.
   `--resume <run_id>` rebuilds the tree. **Test by `kill -9` mid-iteration.** Recovering
   without a human is worth real points.
6. **Search policy.** Draft phase (3 independent drafts from different angles) → greedy improve
   on the best node → debug-first when any node is `buggy` with `repair_attempts < 3` → explore
   the second-best distinct node after 3 flat iterations. Within seed noise (0.0008), prefer
   the simpler node — hidden test is what counts, not validation.
7. **Context assembly.** Where token cost is won or lost. Carries: task card, C's data card,
   parent's full code, parent's metrics, D's top-K ideas, and a **compact** history —
   hypothesis plus metric delta only, never past code. Cap it and log its size.
8. **Accounting.** Cumulative tokens in/out, wall-clock, iterations used of 50. Must be exact —
   it is a reported deliverable and it gates the Feasibility tier.
9. **Final submission step.** At convergence: take the validation-best node, rerun with
   `--split test`, average 3 seeds if time allows, run C's validator, write
   `final/submission.csv`, emit `converged` and `run_end`.

### CLI

```
python -m orchestrator.run --task kuairand-pure --mode {smoke,dev,official} \
       [--max-iters N] [--wall-clock 6h] [--resume RUN_ID] [--seed N]
```

Fully unattended by default. There is no interactive mode. Wanting one is a signal to make the
agent handle the case instead.

### Done when

- [ ] `make check` runs `smoke` (3 iterations, stubbed LLM) in under 60 s, green
- [ ] 50-iteration stub run completes, journal is valid JSONL throughout
- [ ] Convergence passes unit tests on plateau / noise / late-jump
- [ ] `kill -9` then `--resume` continues with no lost nodes
- [ ] A node failing 3 times is marked `dead` and the run routes around it
- [ ] 3 flat iterations trigger an explore action
- [ ] `official` run reaches convergence or the cap with **zero** interventions

### Traps

- A bad node must never poison `best` — only `ok` nodes with a validated submission qualify.
- Errored iterations must not count toward convergence, or three errors in a row declares
  victory at iteration 4.
- Never select the final submission on a single seed. Seed noise is 0.0008; eps is 0.002.
- Never add a human prompt anywhere. Autonomy is directly scored.

---

## B — ML Engineer: Agent Runtime & Sandbox

You own the hands: the LLM calls that write pipeline code, and the sandbox that runs it.
Two scored criteria live here — **Robustness** and **Feasibility**. Own both numbers.

**Files:** `orchestrator/agent.py`, `sandbox.py`, `Makefile`, `.github/workflows/`,
`tests/test_agent.py`, `tests/test_sandbox.py`, `tests/test_faults.py`

You own the prompt *plumbing*; **D owns the prompt text** in `prompts/*.md`. You define the
template variables, D writes the words. Do not edit their content.

Load the `claude-api` skill before writing the client. Do not write it from memory.

### Build order

1. **Sandbox first, LLM second.** It tests for free and it is where Robustness points live.
   Run `python pipeline.py ...` with cwd = the node workspace and a hard timeout (default
   25 min). Capture stdout/stderr to files, keep the last 4000 chars. Parse the single
   `RESULT_JSON {...}` line — missing or malformed is `error_class = "contract"`. Kill the
   whole **process group** on timeout. Record `wall_s` and `peak_rss_mb`. **No network during
   training** — a pipeline that downloads data would breach the no-external-data rule.
2. **Error classification.** stderr → `ErrorClass`, plus the *single most useful* traceback
   slice (deepest frame in `pipeline.py` + the exception line, capped at 1500 chars). Feeding
   the LLM a 200-line traceback is how token budgets die.
3. **Client with accounting.** Every call returns `tokens_in` / `tokens_out` / `model` into the
   `Proposal`. Retry 429/5xx with backoff + jitter; log as a `recovery` event, not an
   intervention.
4. **Structured output.** Force the `Proposal` fields via tool use, not prose parsing.
   `hypothesis` must be non-empty — reject and retry once if it is. That field is what the
   Innovation criterion is scored on.
5. **Repair loop.** Up to **3** attempts per node, each seeing the error excerpt *and* what the
   previous attempt tried. After 3, signal A to mark the node dead. Never loop forever.
6. **Token discipline** — this is scored. Prompt-cache the static block (system + task card +
   data card + library whitelist); it is identical on every call. Never send the dataset, never
   more than the parent's code, never full history. Cap and log the prompt token count.
7. **`make check` and CI.** Lint + unit tests + smoke run, under 60 s, on every push. Keeping
   `main` green is yours to enforce.

### Fault-injection suite — your headline deliverable

Prove each recovers with **zero human input**, and record real results in `STATUS.md`:

| Fault | Class | Expected recovery |
|---|---|---|
| syntax error in generated code | `syntax` | repaired within 3 attempts |
| import outside `requirements-pipeline.txt` | `import` | repaired to an allowed library |
| infinite loop | `timeout` | process group killed, run continues |
| excessive memory | `oom` | killed cleanly, repaired smaller |
| no `submission.csv` written | `contract` | repaired |
| NaN / Inf scores | `eval` | repaired |
| wrong CSV header or row count | `contract` | repaired |
| API 429 / 500 | — | retried with backoff, no iteration lost |

### Done when

- [ ] Every row above passes with zero human input
- [ ] Timeout kills the whole process group; no orphan python processes after a run
- [ ] `hypothesis` is never empty in a real run
- [ ] Journal token counts match the API's reported usage exactly
- [ ] Prompt caching visibly cuts input tokens from the second call onward
- [ ] No secret ever appears in a prompt, log, journal, or node workspace

### Traps

- The LLM returns a whole `pipeline.py`; **we** write it to disk. No diff-application.
- Never retry a repair with the same context, or you get identical broken code three times.
- Do not use temperature 0 for drafts — three identical drafts waste the draft phase. Vary the
  angle in the prompt, not just the seed.
- Watch for the model "winning" by scoring against visible labels or writing a submission for
  the wrong split. Both look like a breakthrough and are both bugs.

---

## C — ML Researcher: Data, Metrics & Evaluation

You own truth. If the metric is subtly wrong, every decision the agent makes all weekend is
based on a lie, and we would not find out until the hidden-test score comes back.

**Files:** `orchestrator/evaluate.py`, `datacard.py`, `report.py`, `data/`,
`requirements-pipeline.txt`, `tests/test_evaluate.py`

### First task, due H+6 — reproduce the official baseline

Nothing else counts until this number is on the board.

1. Unpack `kuairand-starter-kit.zip` (Lark doc §2.4); get KuaiRand-Pure from https://kuairand.com
2. `python3 baseline.py --model fm` — about 40 s on one CPU core
3. Confirm **validation primary ≈ 0.6016** (GAUC 0.6674 / nDCG@5 0.5357)
4. Sanity-check the harness against the published rungs: random → 0.4753, item popularity →
   0.5715. If those two do not reproduce, the harness is wrong.
5. Post the numbers in `STATUS.md`.

**Never reimplement the metrics.** Call the starter kit's `evaluate.py`. The conventions are
pinned: zero-positive users count as nDCG 0 and are *included*; GAUC only over users with
`0 < positives < impressions`, weighted by positive count; gain = `2^rel − 1`.

### Second task, due H+12 — the data card

A markdown summary **under 3000 tokens** that the LLM reads on every draft. It is the agent's
only view of the data, so it sets the quality of every hypothesis D's ideas can produce.
Facts only — the advice belongs in D's idea bank.

- split sizes and date ranges; the `long_view` label and its base rate
- field list with cardinalities, missing rates, user / item / context grouping
- **all 12 feedback signals** and their rates — D's multi-task ideas depend on the agent
  knowing these exist
- video duration distribution and the duration-bias note
- user activity distribution: impressions and positives per user, the 27.1% zero-positive and
  9.2% all-positive users, and what that does to the 0.8645 ceiling
- 3.06% of test rows are repeated `(user_id, video_id)` pairs — `row_id` is the key
- leakage traps: no future data, no test labels, no external data

### Third task — fast iteration

50 iterations in 6 hours means about 7 minutes per pipeline. Make training fast: cache parsed
splits as `.npy`; a shared read-only feature cache keyed by feature-spec hash; `--subsample F`
samples **users, not rows** (row sampling breaks GAUC); keep `requirements-pipeline.txt` tight,
since every library is install time on every node.

### Fourth task, due H+36 — the results generator

`report.py` turns `journal.jsonl` into `RESULTS.md` plus one trajectory PNG. Auto-generated,
never hand-edited — judges can diff the journal against the table.

| Benchmark | Metric | Official baseline | Ours (validation-best) | Absolute delta |
|---|---|---|---|---|
| KuaiRand-Pure | GAUC | 0.6674 | … | … |
| KuaiRand-Pure | nDCG@5 | 0.5357 | … | … |
| KuaiRand-Pure | primary | 0.6016 | … | … |

Plus the resource block (tokens in/out, wall-clock, iterations of 50, GPU-hours) and the
**intervention count**. The PNG is validation primary per iteration with lines at the baseline
(0.6016) and the ceiling (0.8645) and a ±0.0008 noise band. That chart is the project's
headline image — D puts it at the top of the Devpost entry.

### Done when

- [ ] Baseline reproduces within 0.002 of published validation primary
- [ ] Random and item-popularity rungs reproduce (0.4753 / 0.5715)
- [ ] `score()` is deterministic and agrees with `evaluate.py` exactly
- [ ] `validate()` rejects wrong header, row-count mismatch, `row_id` gaps, misalignment,
      non-numeric / NaN / Inf scores
- [ ] Data card is under 3000 tokens and contains no advice
- [ ] A full pipeline run on cached features completes in under 5 minutes
- [ ] `make report RUN=<id>` regenerates RESULTS.md + PNG from the journal alone

### Traps

- **No external training data. Ever.** The one disqualifying rule.
- Do not compute features on train + validation combined during development, or the stopping
  signal is worthless.
- Target encoding must be out-of-fold. This is the single most common way a CTR pipeline scores
  0.75 on validation and 0.59 on hidden test.
- Row sampling breaks GAUC — subsample users.
- `(user_id, video_id)` is not unique. Any join or dedupe assuming uniqueness silently
  misaligns the submission.

---

## D — ML Researcher: Method, Knowledge & Story

You decide what the agent knows and what it thinks to try. This is what makes us a *research*
agent rather than a hyperparameter tuner, and it is scored directly: Innovation & Problem
Insight is judged on **what the agent chose to target and why** — on the choice, not the
implementation.

**Files:** `orchestrator/ideas.yaml`, `knowledge.py`, `prompts/*.md`, `reference/`, `docs/`

You own the prompt **text**; B owns the plumbing around it. Write the words, B wires them.

### First task, due H+8 — the prompts

Three files, plus a system prompt. Keep them in markdown so C and A can read them.

- `draft.md` — cold start. The first draft should aim at **reproducing the official baseline**,
  not at being clever.
- `improve.md` — given the parent's code, its metrics, recent metric deltas and top-K ideas,
  propose **one focused change**. One change per iteration: multi-change proposals make the
  trajectory unreadable and attribution impossible.
- `repair.md` — given code and a classified error excerpt, fix it. Nothing else.
- `system.md` — the standing rules: the library whitelist, the no-external-data rule, the
  pipeline CLI contract, and the demand that `hypothesis` states *why* before *what*.

### Second task, rolling — the idea bank

`orchestrator/ideas.yaml`, entries matching the `Idea` shape in `contracts.md`. **Seeded with 32
entries and 3 dead ends — extend it, do not restart it.**

The tiering follows the organisers' own ranked list of untested directions
(`starter-kit-findings.md`), which **inverts the obvious instinct**. Read that file before adding
anything.

- **T0 — baseline parity and hygiene.** Reproduce FM, early-stop on validation *primary* not
  logloss, seed-average before believing a gain, L2 and LR schedule, item-popularity prior.
- **T1 — objective alignment.** *The organisers' #1 pick and our highest expected payoff.*
  Training is pointwise logloss; the metric is a within-user ranking metric. BPR pairwise,
  listwise softmax over each user's impressions, lambda-weighted pairs, hard negatives,
  within-user centering, margin loss, pointwise/pairwise blends.
- **T2 — user history sequences.** *Organisers' #2, and currently a total blank* — the baseline
  uses no behaviour sequence at all despite hundreds of interactions per user. Mean-pooled
  history, DIN target attention, recency weighting, user × author affinity, short/long split, GRU.
- **T3 — multi-task and watch time.** *Organisers' #3 and #4.* Auxiliary heads on the other
  feedback signals, `play_time_ms` regression, MMoE, PLE, CWM censored watch-time regression,
  duration-conditional debiasing.
- **T4 — architecture, time, ensembling.** *Organisers' #5–#7. Deliberately last.* DeepFM,
  DCN-v2, LightGBM on crosses, rank ensembling, time crosses, recency-weighted training,
  unbiased validation against the random-exposure log, per-user calibration.

**Why architecture is last:** the organisers measured embedding capacity flat (k = 8/16/32 →
0.5895/0.5902/0.5887) and extra static features neutral (0.5940 vs 0.5950). Most teams will
reach for DeepFM on day one. It is the least promising of the five open directions.

### The dead-ends list

`ideas.yaml` also carries a `dead_ends` block — claims that are *measured false*, each with the
number that refutes it. B injects these into every prompt as "do not propose these, and why".
Negative knowledge is worth as much as positive here: each entry saves an iteration spent
rediscovering a published result. Add to it whenever one of our own runs kills an idea.

The sharpest one is structural: **ranking is within-user, so user-side first-order terms
contribute exactly zero** — measured, `item_pop × user_bias` scores identically to plain
`item_pop`. Any idea that adds a user-side feature without an item-side cross is a wasted
iteration. Check every new entry against this before adding it.

Every summary must be **actionable in one iteration** — 2 to 4 sentences an LLM can turn into
code without further research. "Use MMoE" is useless. "Add a shared bottom of two dense layers
with one expert head per feedback signal, train all 12 heads with equal weight, score only the
`long_view` head" is usable.

`retrieve()` takes what has been tried, current best metrics and remaining budget. Simple rules
beat embeddings here: drop tried ids, drop ideas with unmet prerequisites, prefer the lowest
tier with untried entries, escalate a tier when the current one stops producing gains.

### Third task, due H+36 — the reference pipeline

Hand-write a pipeline (T1 features + LightGBM, say) that beats the baseline on validation.
**Not for submission** — it is calibration. It tells us how much headroom exists, so when the
agent plateaus we know whether the bottleneck is the agent or the dataset. Keep it in
`reference/`, out of the agent's context.

### Fourth task, rolling from H+12 — the story

You understand *why* the agent did what it did, so you write it up.

1. **The loop, in one diagram.** Where the LLM sits in read → EDA → features → train → evaluate
   → reflect.
2. **The score.** C's trajectory chart, the absolute delta over baseline, framed against the
   **0.8645 attainable ceiling** — not 1.0. Explain the ceiling; most teams will not, and it
   shows we understood the metric.
3. **What the agent chose to try, and why.** Pull 4–5 of the best hypotheses **verbatim** from
   the journal, with the citation each drew on and the metric delta each produced. Quoting the
   agent is far more convincing than describing it.
4. **How it handled failure.** B's fault-injection table with real results.
5. **What it cost.** Tokens, wall-clock, iterations used, **and the intervention count**.

Plus README reproduction steps and a 3-minute demo video. Draft from H+12 and keep it current.
Teams that start writing at H+60 submit worse projects than they built.

### Done when

- [x] Idea bank has 30+ entries (32 seeded) with a `dead_ends` block
- [ ] Every entry derived from a specific published method carries a citation. Common-practice
      entries (recency weighting, rank ensembling, per-user calibration) legitimately have none —
      leave `citation: null` rather than inventing one
- [ ] Every prompt demands a non-empty `hypothesis` that states *why* before *what*
- [ ] `improve.md` reliably produces one focused change, not a grab-bag
- [ ] Reference pipeline beats the baseline on validation
- [ ] Devpost draft complete at H+48 with placeholders only for final numbers
- [ ] A clean clone plus the README reproduces a `dev` run without asking anyone a question

### Traps

- Ideas the agent cannot implement in one iteration are noise — they burn a turn and produce a
  broken pipeline.
- Do not put advice in C's data card or facts in your idea bank. Facts describe, ideas propose.
- Do not hand-edit `RESULTS.md` or the journal. If a number is wrong, C's generator is wrong.
- Never report hidden-test numbers. We never see them. Report validation and label it.
