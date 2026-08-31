# Architecture

A short technical overview: what runs, in what order, and why the module boundaries are where
they are. The frozen dataclasses and the journal schema live in
[`contracts.md`](../.claude/skills/techjam-track2/references/contracts.md); this document is
about the shape, not the field names.

## The shape

The system is AIDE-like ([arXiv:2502.13138](https://arxiv.org/abs/2502.13138)): **ML
engineering treated as code optimisation, searched over a tree of solution programs.** Each
node in the tree is one complete, self-contained `pipeline.py`. Improving a solution means
asking an LLM for a new version of the whole file; debugging one means asking for a corrected
version of the whole file. There is no diff format, no patch application, no plugin registry.

That choice buys three things. A node is trivial to sandbox, because it is one file with a
fixed CLI. A node is trivial to revert, because nothing outside its own directory changed. And
the LLM can hold the entire artefact in context, so it never reasons about code it cannot see.

## Data flow

```
tasks/<name>.yaml ─────► TaskConfig ─────► every box below reads its facts from here
                              │
                              ▼
                    ┌───────────────────┐
   ideas.yaml ─────►│                   │
   (knowledge.py)   │   Context         │──────► agent.py ────► Proposal
                    │   • task spec     │        (one LLM       • hypothesis
   datacard.py ────►│   • data card     │         call per      • plan
   (profile.py)     │   • parent code   │         node)         • code
                    │   • top-k ideas   │                       • idea_ids
   core.py ────────►│   • history       │                            │
   (tree, budget)   │   • budget left   │                            ▼
                    └───────────────────┘                    pipeline.py on disk
                              ▲                                      │
                              │                                      ▼
                              │                              ┌──────────────┐
                              │            ExecResult ◄──────│  sandbox.py  │
                              │            • ok / error_class│  subprocess, │
                              │            • stdout tail     │  timeout,    │
                              │            • submission path │  no network  │
                              │                   │          └──────────────┘
                              │                   ▼
                              │           evaluate.validate()  ── contract error ──┐
                              │                   │                                │
                              │                   ▼                                │
                              │           evaluate.score()  ──► metrics            │
                              │                   │                                │
                    ┌─────────┴─────────┐         ▼                                │
                    │      core.py      │◄── update best, check convergence        │
                    │  loop + policy.py │                                          │
                    └─────────┬─────────┘◄─────────────────────────────────────────┘
                              │                  repair (up to 3x), then route around
                              ▼
                     journal.py ──► journal.jsonl ──► report.py ──► RESULTS.md + trajectory.png
```

One iteration, concretely:

1. `policy.next_action(tree)` returns a `kind` (draft / improve / debug) and a parent node,
   plus a one-line *reason* that is journalled — a judge can read why the search moved where
   it did.
2. `core.py` assembles a `Context`: the task spec, the data card, the parent's code and
   metrics, the top-k ideas from `knowledge.retrieve()`, a compact history (hypothesis and
   metric delta only, never past code), and the remaining iterations / seconds / tokens.
3. `agent.draft|improve|repair(ctx, …)` makes **one** LLM call and returns a `Proposal`.
4. The proposal's code is written to `runs/<run_id>/nodes/nNNN/pipeline.py`.
5. `sandbox.run(node, split="val", …)` executes it in a subprocess with a timeout, a killed
   process group on expiry, and outbound sockets blocked.
6. On failure: journal the error, and schedule a `debug` node against the same parent. After
   three repair attempts the node is marked `dead` and the search routes around it.
7. On success: `evaluate.validate()` checks the submission against the task's schema and row
   alignment; a rejection becomes `error_class="contract"` and goes back through repair.
   Otherwise `evaluate.score()` produces the metrics, `best` is updated, and convergence is
   checked.
8. Every step emits a line to `journal.jsonl`, flushed immediately.

The run stops at whichever comes first: the iteration cap, the wall-clock ceiling, the token
budget, or convergence — validation primary not improving by more than `eps = 0.002` over
`n = 3` **scored** iterations. Failed iterations do not count toward `n`, because a run that
crashed three times has not converged, it has stalled.

Finalisation reruns the validation-best node with `--split test`, once, across three seeds in
`official` mode, and validates the result without ever reading a test label.

## Why the boundaries are where they are

### `contracts.py` — frozen dataclasses, imports nothing

Four people were building four modules simultaneously against each other's unfinished code.
The only way that works is if the shapes are agreed first and never quietly changed. So
`contracts.py` holds every shared dataclass, imports nothing else from `orchestrator/`, and
is frozen: changing it requires a written proposal, a backward-compatible shape, and an ack.
Everything else is duck-typed against those shapes, which is what lets `core.py` run
identically against the stubs and against the real modules.

### The four seams — `agent`, `sandbox`, `evaluate`, `knowledge`

Each is a module-level function signature, not a class hierarchy, and each has a stub in
`tests/stubs/`. `run.py` resolves them per seam: `auto` imports the real module if it exists
and exposes its functions, and falls back to the stub if it does not. `smoke` mode pins all
three to stubs.

The point was never elegance — it was that nobody could ever be blocked on somebody else's
file, and that CI has a full end-to-end run it can execute on every push with no API key, no
dataset and no spend. That property survived into the finished system: the quickstart in the
README is the same path CI takes.

### `policy.py` — pure functions over the tree

No I/O, no clock, no LLM. The policy decides where 50 expensive iterations go, and it is the
component most likely to be subtly wrong: a greedy search that only ever improves the best
node will polish one dead end for six hours. Making it pure is what makes it unit-testable
without a network or a dataset. Its priority order is:

1. **Debug first.** Any leaf that failed and still has repair attempts left. A broken program
   is the cheapest thing in the tree to make valuable.
2. **Draft phase.** Until *n* independent drafts exist, keep drafting from a different angle.
   Independent drafts are the only cheap protection against a bad first program anchoring the
   entire search.
3. **Rescue.** Nothing scored and nothing repairable: draft again.
4. **Explore.** After several flat scored iterations, improve the second-best *distinct* node
   rather than the best one.
5. **Greedy improve** on the best node.

Ties inside seed noise go to the *simpler* node. Validation is not the score we are ranked on,
so between two statistically indistinguishable programs the simpler one is the better bet on
hidden test.

### `agent.py` vs `prompts/*.md` — plumbing and words are owned separately

`agent.py` defines the template variables and substitutes them; the prompt text lives in
markdown files. This is not decoration: prompt wording is the highest-leverage,
fastest-changing part of the system, and making it a code change would have serialised it
behind whoever owned the runtime. It also makes the prompts reviewable as prose.

The prompts are templated per task (`$submission_header`, `$prediction_column`,
`$order_note`, `$dead_ends`, …), so one set of files serves any dataset. The static block —
system prompt, task card, data card, library whitelist — is identical on every call in a run
and is marked for prompt caching, so from the second call it bills as a cache read.

### `evaluate.py` — a wrapper, never a reimplementation

The metrics are the organisers'. `score()` delegates to `vendor/starter_kit/evaluate.py`,
which is the sole authority on the conventions that are easy to get subtly wrong: zero-positive
users count as nDCG 0 and are *included* in the mean; GAUC covers only users with
`0 < positives < impressions`, weighted by positive count; gain is `2^rel − 1`. A
reimplementation that differs in any of those makes every decision the agent takes afterwards
wrong, silently, and in a direction nobody would notice until the hidden-test score came back.

`validate()` mirrors `submit.py::read_submission` check for check, in the same order, but
returns `(ok, message)` instead of raising — because the orchestrator needs to turn a bad
submission into a repairable error rather than a crashed run.

### The hidden-test guard is mechanical, not a promise

KuaiRand-Pure ships the test labels. "Hidden test" means the organisers score their own copy;
it does not mean we are unable to read ours. An LLM writing its own pipeline will find those
labels if nothing stops it, and tuning on test is the single easiest way for an agent to look
brilliant and score badly.

So the rule is enforced in three places rather than asserted once:
`evaluate.score(..., "test")` refuses to run unless `ALLOW_TEST_SCORING=1` is set explicitly;
scored iterations only ever run `--split val`; and for non-KuaiRand tasks the materialised
`test.csv` is written with the target column removed, so no-peeking is enforced by absence.
The system prompt states the rule as well, but the prompt is the weakest of the four.

### `taskspec.py` and `tasks/*.yaml` — the dataset is configuration

Originally the dataset was constants scattered through the code: the submission header in the
sandbox, the metric names in the evaluator, the row order in the loader, KuaiRand prose in the
system prompt. Pulling all of it into one YAML file per problem is what makes the claim "give
it a dataset and a problem statement" true rather than aspirational — and it exposed a real
bug in the process, because rendering the system prompt without task substitution had been
quietly telling every task to write KuaiRand's submission header.

The rule that keeps this boundary honest: **a task file describes, it never instructs.** No
hyperparameters, no model suggestions. What to try is the agent's job.

### `datacard.py` / `profile.py` — facts only, no advice

The LLM never sees the CSVs; sending them would blow the token budget that Feasibility is
scored on. It sees a generated markdown card instead — split sizes, column types,
cardinalities, missingness, relationship to target, and warnings for constant, identifier-like
and leakage-grade columns.

The card contains no recommendations, deliberately. What to *try* belongs in the idea bank,
where every attempt can be attributed to an idea id in the journal. A suggestion that leaks
into the data card is unattributable and biases every proposal the agent ever makes.

### `knowledge.py` / `ideas.yaml` — rule-based retrieval, and negative knowledge

Retrieval is rules, not embeddings: 33 curated ideas and one call per iteration do not need a
vector store, and adding one would add a dependency, a failure mode and a token cost to solve
a problem we do not have. The rules are: drop what was tried, drop what has unmet
prerequisites, drop what is too slow to finish when the budget is nearly spent, serve the
lowest tier that still has entries, and include one idea from the next tier as a lookahead so
escalation is a gradient rather than a cliff.

Two design points are worth naming.

**The tier order is the organisers' ranked list, not the obvious one.** Ranking losses are
T1; user-history sequences T2; multi-task and watch-time T3; *architecture swaps are T4,
last*. That inverts most people's instinct to reach for DeepFM first, and it is inverted on
purpose: capacity and feature count were both measured flat on this dataset, so a bigger model
is the least promising of the open directions.

**Dead ends are first-class.** Four measured-false claims ride in the cached system block on
every call, each with the number that refutes it. Negative knowledge is worth as much as
positive here — each of these would otherwise cost an iteration to rediscover. They are loaded
from *the task's own* bank, so a second task can never inherit KuaiRand's conclusions.

### `journal.py` — the log is a deliverable, not a log

`journal.jsonl` is read by judges to score Autonomy, Robustness and Innovation. So: one valid
single-line JSON object per line, always; flushed on every write so a `kill -9` loses nothing;
never a secret; never a raw traceback past the cap. `report.py` reads that file and nothing
else, which means `RESULTS.md` can be regenerated at any moment — mid-run, or after a crash —
and a run that dies at hour four still has a current deliverable. Malformed lines are counted
and skipped rather than raised, because a reporting bug must never be why a finished run has
no report.

## What we deliberately did not build

- **A web UI or dashboard.** Nobody scores it. `RESULTS.md` plus one PNG is the deliverable.
- **Planner / coder / critic personas inside the agent.** One well-prompted call per node.
  Multiple personas burn scored tokens and add failure modes to buy an effect we could not
  measure.
- **A plugin system on top of the task layer.** `tasks/*.yaml` is where genericity stops.
- **GPU or distributed training.** 100 iterations of the official baseline is roughly 28
  minutes of one CPU core. Wall-clock is scored; a GPU would not help and might hurt.
- **A metric implementation of our own.** See above.
