# Role B — Agent Runtime and Execution Sandbox

You own the hands: the LLM calls that write pipeline code, and the sandbox that runs it.
Two judged criteria live almost entirely in your code — **Robustness** (how failures are
handled) and **Feasibility** (token spend). Own both numbers.

## You own

```
orchestrator/agent/client.py      Anthropic API client, retries, token accounting
orchestrator/agent/prompts/       draft.md, improve.md, repair.md, system.md
orchestrator/agent/parse.py       response -> Proposal, code extraction, validation
orchestrator/agent/agent.py       draft() / improve() / repair()
orchestrator/exec/sandbox.py      subprocess runner: timeout, limits, capture
orchestrator/exec/classify.py     stderr -> ErrorClass + minimal useful excerpt
tests/test_parse.py, tests/test_sandbox.py, tests/test_faults.py
```

## You do not touch

`orchestrator/core/`, `orchestrator/search/` (A) · `orchestrator/eval/`,
`orchestrator/knowledge/` (C) · `orchestrator/report/`, `docs/` (D).

## Before you write the client

Load the `claude-api` skill. Do not write the Anthropic client from memory — get the model
ids, streaming, tool-use and prompt-caching details from the skill.

## Build order

1. **Sandbox first, LLM second.** The sandbox is the part that can be tested without spending
   a cent, and it is where Robustness points live.
   - run `python pipeline.py --data-dir ... --out-dir ... --split val --seed 0` in the node
     workspace as cwd, with a hard timeout (default 25 min, configurable per mode)
   - capture stdout/stderr to files, keep only the last 4000 chars in `ExecResult`
   - parse the single `RESULT_JSON {...}` line; a missing or malformed one is
     `error_class = "contract"`
   - kill the whole process group on timeout, not just the parent
   - record `wall_s` and `peak_rss_mb`
   - no network access during training; the only network the run needs is your API calls
2. **Error classification.** Map stderr to `ErrorClass` and extract the *single most useful*
   slice of the traceback (the deepest frame in `pipeline.py` plus the exception line, capped
   at 1500 chars). Feeding the LLM a 200-line traceback is how token budgets die.
   Classes: `syntax`, `import`, `data`, `runtime`, `oom`, `timeout`, `contract`, `eval`, `unknown`.
3. **Client with accounting.** Every call returns `tokens_in` / `tokens_out` / `model` and
   they flow into the `Proposal`. Retry on 429/5xx with exponential backoff and jitter; a
   retry is not an intervention, but log it as a `recovery` event.
4. **Prompts.** Three of them, kept in markdown files so C and D can read them:
   - `draft.md` — cold start. Given the task card, data card and top ideas, write a complete
     `pipeline.py`. First draft should aim at *reproducing the official baseline*, not at
     being clever.
   - `improve.md` — given the parent's full code, its metrics, the metric deltas of recent
     attempts, and top-K ideas, propose one focused change. **One change per iteration.**
     Multi-change proposals make the trajectory unreadable and the attribution impossible.
   - `repair.md` — given the code and the classified error excerpt, fix it. Nothing else.
5. **Structured output.** Force the model to return the `Proposal` fields. Use tool-use /
   structured output rather than parsing prose. `hypothesis` must be non-empty — reject and
   retry once if it is, because that field is what the Innovation criterion is scored on.
6. **Repair loop.** On a failed node: up to **3** repair attempts. Each repair sees the error
   excerpt and the previous fix attempt. After 3, return a signal so A marks the node dead.
   Never loop forever, never silently swallow.
7. **Token discipline.** This is scored.
   - prompt-cache the static block (system prompt + task card + data card + requirements
     whitelist) — it is identical across every call
   - never send the dataset, never send more than the parent's code, never send full history
   - cap the assembled prompt and log its token count every call
   - use a cheaper model for `repair` if quality allows; measure before deciding

## Fault-injection suite (this is your headline deliverable)

Build `tests/test_faults.py` with a fixture pipeline for each fault, and prove the system
recovers **with zero human input**:

| Fault | Expected class | Expected recovery |
|---|---|---|
| syntax error in generated code | `syntax` | repair succeeds within 3 attempts |
| import of a package not in `requirements-pipeline.txt` | `import` | repair falls back to an allowed library |
| infinite loop | `timeout` | process group killed, node marked, run continues |
| allocates far too much memory | `oom` | killed cleanly, classified, repaired to a smaller footprint |
| writes no `submission.csv` | `contract` | repaired |
| writes NaN / Inf scores | `eval` | repaired |
| wrong CSV header or row count | `contract` | repaired |
| API returns 429 / 500 | n/a | retried with backoff, logged as `recovery`, no iteration lost |

Record this table with real results in `STATUS.md`. D will put it in the Devpost writeup —
it is the cleanest possible evidence for the Robustness criterion.

## Acceptance tests — you are done when

- [ ] Every row of the fault table passes with zero human input
- [ ] Timeout kills the entire process group; no orphan python processes after a run
- [ ] `Proposal.hypothesis` is never empty in a real run
- [ ] Token counts in the journal match the API's reported usage exactly
- [ ] Prompt caching demonstrably cuts input tokens on the second call onward
- [ ] Secrets never appear in prompts, logs, journal, or the node workspaces

## Traps

- **Do not let the LLM edit files directly.** It returns a whole `pipeline.py`; we write it.
  Diff-application is a failure mode we do not need this weekend.
- **Do not give the sandbox network access during training.** A generated pipeline that
  downloads data would breach the no-external-data rule. Block it and make the block explicit
  in the prompt.
- **Do not retry a repair with the same context.** Include what the previous attempt tried and
  why it failed, or you will get the identical broken code three times.
- **Do not use temperature 0 for drafts.** Three identical drafts waste the draft phase. Vary
  the angle in the prompt, not just the seed.
- Watch out for the model "solving" the task by scoring against the labels it can see, or by
  writing a submission for the wrong split. Both look like a huge win and are both bugs.
