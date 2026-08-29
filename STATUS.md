# STATUS

Shared scratchpad. Update your section before every sync (H+6, H+12, H+24, H+36, H+42, H+48,
H+60). Bullets, not prose.

## Key numbers

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline — validation | 0.6674 | 0.5357 | **0.6016** |
| Official baseline — hidden test | 0.6610 | 0.5282 | **0.5946** |
| Our reproduction (C fills) | — | — | — |
| Our best autonomous run (validation) | — | — | — |
| Attainable ceiling | 1.0000 | 0.7289 | 0.8645 |

## A — ML Engineer: Orchestrator & Run

- [ ] `contracts.py` + `journal.py` + stubs frozen (H+2 — everyone is blocked on this)
- [ ] loop, tree, convergence, budget
- [ ] `kill -9` then `--resume` verified
- now:
- blocked on:

## B — ML Engineer: Agent Runtime & Sandbox

- [x] sandbox runs arbitrary `pipeline.py` with timeout + process-group kill
- [x] repair loop with error classification
- [x] fault-injection table passing — 13/13 recover, table below
- [x] `make check` in CI — lint + 98 tests + smoke, 15 s local
- now: `agent.py` + `sandbox.py` + `Makefile` + `.github/workflows/ci.yml` landed on
  `feat/b-sandbox-agent`. Both function-level seams from contracts.md §3 are real:
  `sandbox.run(node, split=, seed=, timeout_s=, subsample=)` and
  `Agent.draft/improve/repair(ctx, ...)`. `sandbox.run` also takes `data_dir=`,
  `mem_limit_mb=`, `python=`, `allow_network=` — all keyword-only with defaults, so
  the frozen signature still holds.
- [x] provider-agnostic client: Anthropic (default) + OpenAI adapter + stub, chosen
  by which key is present. See `## Contract change proposed` — needs one ack.
- [x] `.env` loading, key-shape guard, `make models`
- **Decision taken: we run on OpenAI.** No Anthropic key is available to the team.
  Anthropic stays supported and stays the code default; `TECHJAM_LLM=openai` in `.env`
  selects the provider for our runs. Contract change below still needs one ack.
- [x] live API probe — key works, 120 models reachable. **Three findings that change
  how we configure the run:**
  1. **`gpt-5.5` rejects any `temperature` except 1** (400, and a 400 is deliberately
     not retryable). The adapter would have raised on the *first* `improve()` call of
     the scored run — temperature 0.6 — and killed that node, having looked perfectly
     healthy for exactly one iteration. Now caught, dropped, retried once and
     remembered per model. Costs us nothing real: draft diversity comes from the
     prompt angle, not temperature, which is what roles.md asks for anyway.
  2. **Codex models are not served by `v1/chat/completions`** — `gpt-5.3-codex` and
     friends 404 with "use the v1/responses endpoint". Out of scope unless someone
     writes a Responses adapter. Do not put one in `TECHJAM_MODEL`.
  3. **`gpt-5.4` accepts a custom temperature; `gpt-5.5` does not.** Both work.
- [x] **live shakedown done — everything in my "Done when" list is now verified
  against the real API, not the stub.**
  - **Prompt caching works.** `gpt-5.4` cold call: 0 cached. Every call after:
    1,536 of 2,080 input tokens served from cache, ~74%. Our static block leads the
    request and is byte-identical across calls, which is exactly what OpenAI's
    automatic prefix caching needs. No `cache_control` marker required.
  - **Token accounting is exact.** `Usage.tokens_in` equals the API's own
    `prompt_tokens` on every call. A journalled total will match the bill.
  - **`hypothesis` is never empty**, and the ones we got read like real claims that
    could turn out false — "repeated user_id/video_id identities with leakage-safe
    target statistics", "categorical ids in validation not observed in training".
    D: these are quotable for the Innovation section.
  - **Draft diversity 3/3 distinct** on both models, from prompt angle alone.
  - **Temperature fallback verified live on the call that would have broken.**
    `improve()` at 0.6 against `gpt-5.5`: fallback engaged, call succeeded,
    `fixed_temperature_models == {'gpt-5.5'}`. The draft-only shakedown passed
    *without* triggering it — drafts run at 1.0, the one value that model accepts —
    so a lighter smoke test would have declared this healthy and lost the run.

| 3 live drafts | wall-clock | output tokens | input (cached) |
|---|---|---|---|
| `gpt-5.5` | 160 s | 16,530 | 6,242 (4,608) |
| `gpt-5.4` | 66 s | 10,973 | 6,242 (3,072) |

- **`TECHJAM_MODEL=gpt-5.5`**, set in `.env`. Capability first: the Feasibility tier
  is *only scored among entries that beat the baseline*, so the gate is the score,
  not the cost. `gpt-5.4` is the documented fallback — 2.4x faster and ~1.5x cheaper
  in output tokens — and switching is one env var, no code change. If wall-clock gets
  tight, note that LLM time is ~50 s/iteration on 5.5 vs ~22 s on 5.4, which is ~25
  minutes across a 50-iteration run; training still dominates, but the 6 h ceiling is
  not roomy.
- next: token cap enforcement in `Context`; a cheaper model for `repair` calls only
  (they are mechanical) is an easy Feasibility win if we need one later.

**Everyone: how to configure your box.** `.env` at the repo root, gitignored, never
committed. Nothing else reads a key.

```
TECHJAM_LLM=openai
OPENAI_API_KEY=sk-proj-...
TECHJAM_MODEL=<pick one from `make models`>
```

`make models` lists what your key can actually reach and prints ids only, never the
key. `TECHJAM_MODEL` is mandatory on OpenAI — see the contract note. `make check` and
`make smoke` need none of this: they run stubbed, with no key and no spend.
- blocked on: nothing. A's `run.py` gates the real `make smoke` (it currently reports
  PENDING rather than failing, so `main` stays green); C's `requirements-pipeline.txt`
  gates the library whitelist the system prompt states; D's `prompts/*.md` override
  the placeholders automatically the moment they land — no wiring needed.

**For A — what you can rely on now:**
- `sandbox.run()` never raises for a bad pipeline; every failure is an `ExecResult`
  with `ok=False` and an `error_class`. It raises only if `pipeline.py` is missing,
  which is an orchestrator bug, not an agent one.
- `agent.repair_exhausted(node)` is the "mark it dead" predicate (3 attempts).
- `Agent(on_usage=..., on_recovery=...)` — wire these to `journal.emit`. API retries
  arrive as `recovery` events, never as interventions.
- `agent.Usage.tokens_in` counts cache writes and reads as well as plain input, so
  the reported total matches what we are actually billed.
- Set `TECHJAM_LLM=stub` for `smoke`: no key, no spend, and the stub pipeline is
  contract-valid, so the whole loop exercises for real.

## C — ML Researcher: Data, Metrics & Evaluation

- [ ] official baseline reproduced
- [ ] random / item-popularity rungs reproduce (0.4753 / 0.5715)
- [ ] data card under 3000 tokens
- [ ] `report.py` → RESULTS.md + trajectory PNG
- now:
- blocked on:

## D — ML Researcher: Method, Knowledge & Story

- [ ] `system.md` + `draft.md` + `improve.md` + `repair.md`
- [ ] idea bank entries: 0 / 30
- [ ] reference pipeline beats baseline
- [ ] Devpost draft
- now:
- blocked on:

## Requests

Cross-team asks. Format: `A -> C: need X because Y`.

- **B -> A: `orchestrator/contracts.py` is a placeholder I transcribed, please
  overwrite it.** Nothing else existed at H+0 and `sandbox.py`/`agent.py` import it.
  The dataclasses are a verbatim copy of contracts.md §2 — no design decisions taken,
  so your version should drop straight in. Delete my header comment when you do.
- **B -> A: `Context` needs to be a real dataclass — please ack or replace.**
  contracts.md §3 specifies it in prose only, and `agent.py` cannot be written against
  prose. I added `Context`, `HistoryEntry` and `Budget` at the bottom of `contracts.py`,
  clearly marked. Fields: `task`, `data_card`, `ideas`, `history`, `budget`,
  `parent_code`, `parent_metrics`, `library_whitelist`, `run_id`, `iteration`,
  `draft_angle`. `history` carries hypothesis + metric delta only — never code, which
  is the single biggest lever on the token bill. Change the shape if you like; tell me
  and I will follow.
- **B -> C: `requirements-pipeline.txt` please, even if it is just `numpy`.**
  It is the library whitelist the system prompt states, and `import` errors are
  repaired against it. Until it exists the agent is told "stdlib and numpy only".
- **B -> C: does `evaluate.validate()` distinguish `contract` from `eval` failures?**
  The sandbox already rejects a wrong header, a bad row count and NaN/Inf scores
  before your evaluator sees the file, so we should not disagree about which class a
  given breakage is. Structural checks only — no scoring happens in `sandbox.py`.
- **B -> D: `prompts/{system,draft,improve,repair}.md` are wired and will be picked up
  automatically.** Template variables available to you, `$name` style, unknown names
  are left alone rather than crashing a run: `$run_id $iteration $budget $ideas
  $history $draft_angle $parent_code $parent_metrics $parent_node_id $error_class
  $error_excerpt $stdout_tail $attempt $max_attempts $previous_attempts`. The
  placeholders in `agent.py` say PLACEHOLDER out loud so nobody mistakes them for
  your work. `hypothesis`, `plan`, `code`, `idea_ids` are forced through tool use, so
  the prompts do not need to specify an output format.

## Contract change proposed

Format: what changes, why, who acked, when the old shape can go.

- **B: contracts.md §8 says "`ANTHROPIC_API_KEY` from env only". Widen it to
  "`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` from env only". Needs one ack.**
  - *What changes:* nothing in code that anyone else touches. `agent.py` speaks one
    shape — Anthropic's Messages API — and an `OpenAIClient` adapter translates. The
    orchestrator never learns which provider is live. `make_client()` prefers an
    Anthropic key, falls back to an OpenAI one, and `TECHJAM_LLM=anthropic|openai|stub`
    overrides. The old shape keeps working unchanged and stays the default.
  - *Why:* the key we have on the box right now is an OpenAI one, and being able to
    switch providers without a code change is also a Robustness hedge — if one starts
    throttling us during the 6-hour scored run, we move rather than lose the run.
  - *One thing A must know for the accounting:* Anthropic reports `input_tokens`
    *excluding* cache reads, so billed input is the sum of three fields. OpenAI's
    `prompt_tokens` *includes* `cached_tokens`. Summing both the same way would
    double-count every cached token and overstate a scored deliverable. `agent.py`
    normalises at the adapter, so `Usage.tokens_in` equals what we are billed on
    either provider — journal it directly and do not re-derive it.
  - *Model id:* on Anthropic it defaults to `claude-opus-5`. On OpenAI, `TECHJAM_MODEL`
    is **required** — `default_model_for()` raises rather than guess an id, because a
    wrong one discovered four hours into a scored run is a lost run.
  - *Acked by:* — (needs one; B has taken the OpenAI path in the meantime because it
    is the only key the team has, and the Anthropic default is untouched)
  - *Old shape can go:* never; both stay.

## Fault-injection results (B fills, D uses in the writeup)

Measured, not asserted: `tests/test_faults.py` runs each fault as a real subprocess on
every push. Regenerate with `make test`.

| Fault | Class | Recovered | Attempts | Detected in |
|---|---|---|---|---|
| syntax error in generated code | `syntax` | yes | 1 | 0.2 s |
| import outside the whitelist | `import` | yes | 1 | 0.2 s |
| infinite loop | `timeout` | yes | 1 | 2.2 s |
| orphaned grandchild process | `timeout` | yes | 1 | 2.2 s |
| excessive memory | `oom` | yes | 1 | 1.1 s |
| no `submission.csv` written | `contract` | yes | 1 | 0.2 s |
| no `RESULT_JSON` line | `contract` | yes | 1 | 0.2 s |
| wrong CSV header | `contract` | yes | 1 | 0.2 s |
| row count disagrees with `RESULT_JSON` | `contract` | yes | 1 | 0.2 s |
| NaN / Inf scores | `eval` | yes | 1 | 0.2 s |
| external data download | `data` | yes | 1 | 0.2 s |
| pipeline prompts for input | `runtime` | yes | 1 | 0.2 s |
| uncaught exception mid-training | `runtime` | yes | 1 | 0.2 s |
| API 429 / 500 | — | yes | 2 retries | — |

**Read the Attempts column honestly.** Detection, classification, killing and routing
are real. The *fix* comes from a scripted agent that returns working code first time,
so "1" measures the plumbing, not the model. Real attempt counts come from the first
`official` run and this table gets regenerated from its journal — D, do not quote the
current numbers as the agent's repair rate.

Also verified: the timeout kill takes the whole process group (an orphaned grandchild
would eat a core for the rest of a six-hour run); no secret-shaped environment variable
reaches a node workspace; a pipeline that tries to download external data is blocked at
the socket, which is the one disqualifying rule.
