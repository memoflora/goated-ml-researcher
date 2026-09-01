# goated-ml-researcher — Devpost writeup (draft)

**TikTok TechJam 2026 · Track 2 — Autonomous ML Research Agent for Recommender Systems**

> **Status: draft.** Everything that depends on the scored run is marked `TODO`. No number
> below is invented: either it was measured and is stated with its source, or it is a
> placeholder. Placeholders marked `TODO: quote verbatim from journal` must be filled by
> copying the agent's own words out of `runs/<run_id>/journal.jsonl` — never paraphrased,
> never written for it.

---

## What it is

An autonomous ML research engineer. You give it a dataset and a problem statement; it writes
the training pipeline, runs it, reads the score, forms a hypothesis about what is limiting it,
writes a new pipeline, and repeats — for up to 50 iterations or 6 hours, with nobody watching.

Every solution it produces is one self-contained `pipeline.py`. Improving means writing a new
one. Debugging means writing a corrected one. The search is over a tree of programs
(AIDE-shaped, [arXiv:2502.13138](https://arxiv.org/abs/2502.13138)), and every hypothesis,
failure, repair and score is written to an append-only journal as it happens.

## The loop

```
                    ┌──────────────────────────────────────────────┐
                    │  tasks/<name>.yaml — the whole problem spec   │
                    │  data · target · split · metrics · schema    │
                    └───────────────────────┬──────────────────────┘
                                            │  everything below reads its facts from here
                                            ▼
   ┌─────────────┐   what to try next   ┌────────┐   context   ┌──────────────────┐
   │  policy     │─────────────────────►│  core  │────────────►│  LLM agent       │
   │ debug-first │  draft/improve/debug │  loop  │  task card  │  ONE call        │
   │ draft phase │                      │  tree  │  parent code│  hypothesis      │
   │ explore     │◄─────────────────────│ budget │  top-k ideas│  plan            │
   │ improve     │      updated tree    └───┬────┘  history    │  full pipeline.py│
   └─────────────┘                          │                  └────────┬─────────┘
          ▲                                 │                           │
          │                                 │                           ▼
          │                        ┌────────┴────────┐         ┌────────────────┐
          │                        │  idea bank      │         │  sandbox       │
          │                        │  32 cited ideas │         │  subprocess    │
          │                        │  5 tiers        │         │  timeout, kill │
          │                        │  4 dead ends    │         │  no network    │
          │                        └─────────────────┘         └───────┬────────┘
          │                                                            │
          │                                                            ▼
          │                                                   ┌────────────────┐
          │        failed: repair ≤3x, then dead, route around │  evaluate      │
          └───────────────────────────────────────────────────│  validate()    │
                                 scored: update best,          │  score()       │
                                 check convergence             └───────┬────────┘
                                                                       │
                                                                       ▼
                                            journal.jsonl ──► RESULTS.md + trajectory.png
```

Stop conditions, whichever comes first: 50 iterations, 6 hours wall-clock, the token budget,
or convergence — validation primary not improving by more than `eps = 0.002` over `N = 3`
**scored** iterations. Failed iterations do not count toward `N`; a run that crashed three
times has not converged, it has stalled.

The scored submission is the validation-best checkpoint, retrained once on train+validation
and scored once on hidden test.

## The score, against the right denominator

**Do not read this against 1.0.** A *perfect* ranking of the hidden test scores 0.8645, not
1.0, because 27.1% of test users have no positive label — their nDCG is 0 for any model that
has ever existed. Random scoring sits at 0.4753. So the whole attainable range is 0.4753 →
0.8645, and the official baseline at 0.5946 already occupies roughly a third of it.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Random — hidden test | 0.4996 | 0.4511 | 0.4753 |
| Item popularity — hidden test | 0.6308 | 0.5121 | 0.5715 |
| **Official baseline — hidden test** | 0.6610 | 0.5282 | **0.5946** |
| Official baseline — validation | 0.6674 | 0.5357 | 0.6016 |
| Our reproduction — validation | 0.6671 | 0.5358 | 0.6015 |
| BPR-FM reference (our calibration control) | 0.6698 | 0.5365 | 0.6032 |
| Oracle ceiling — validation | 1.0000 | 0.6968 | 0.8484 |
| **Oracle ceiling — hidden test** | 1.0000 | 0.7289 | **0.8645** |

`primary` is the equal-weighted mean of GAUC and nDCG@5.

**Our result.**

| | GAUC | nDCG@5 | primary | absolute delta vs baseline |
|---|---|---|---|---|
| Validation-best checkpoint | `TODO` | `TODO` | `TODO` | `TODO` |
| Hidden test (scored once) | `TODO` | `TODO` | `TODO` | `TODO` |

`TODO: fill from runs/<run_id>/RESULTS.md once the official run has finished.`

Framing to use when those numbers land: validation headroom above the baseline is
0.8484 − 0.6016 = **0.2468**. For scale, our own controlled ranking-loss experiment captured
about 0.6% of that headroom with one change. **Gains here come from stacking, not from one
clever idea**, which is exactly why an agent that can run 50 disciplined single-variable
iterations is the right tool and a human with three good ideas is not.

## What the agent chose to try, and why

This is the part we care about most, and it is the part we refuse to write on the agent's
behalf. Every proposal the agent makes begins with a hypothesis — what it believes is
currently limiting the score, what it is changing, and how much it expects that to move — and
every one of those paragraphs is in the journal, timestamped, before the result was known.

`TODO: quote verbatim from journal — the first draft's hypothesis (event=proposal,
kind=draft, iteration 1).`

`TODO: quote verbatim from journal — the hypothesis behind the first improvement that
actually moved the score, with the delta it produced.`

`TODO: quote verbatim from journal — one hypothesis that was WRONG, and what the agent said
in the next iteration after reading the result. A search that never proposes a losing idea is
not searching; what matters is whether it noticed.`

`TODO: quote verbatim from journal — any proposal the agent made that was NOT in the idea
bank (empty idea_ids), with its stated reason. This is the strongest possible evidence for
Innovation and it may not happen; if it does not, say so rather than dressing up a retrieval
hit as originality.`

### What we put in front of it, and why that ordering is unusual

The agent draws on a hand-built bank of **33 ideas in five tiers**, 17 of them citing the
paper they come from, plus **four measured dead ends** that ride in every prompt as "do not
propose these, and here is the number that refutes them". The tiering deliberately inverts the
obvious instinct:

| Tier | Direction | Why there |
|---|---|---|
| T0 | Baseline parity and hygiene | Cheap, must-do, small payoff. Nothing is measurable until something reproducible scores. |
| T1 | **Objective alignment** — BPR, listwise softmax, LambdaRank-style weighting | Training optimises a *pointwise* logloss; the metric is a *within-user ranking* metric. This is the organisers' own #1 open direction. |
| T2 | User history sequences — mean pooling, DIN target attention, SIM-style splits | Every user has hundreds to thousands of train interactions and the baseline looks at none of them. |
| T3 | Multi-task and watch time — auxiliary heads, MMoE, PLE, censored watch-time regression | The logs carry 12 feedback signals; only `long_view` is scored. |
| T4 | **Architecture** — DeepFM, DCN-v2, GBDT blends, ensembling | *Last, on purpose.* |

Most teams will reach for DeepFM first. We put it last, because it was measured not to help:
embedding k = 8 / 16 / 32 gives 0.5895 / 0.5902 / 0.5887 — flat. Adding all 13 available
feature fields scores 0.5940 against the 5-field baseline's 0.5950 — inside noise, slightly
worse. **The bottleneck on this dataset is neither capacity nor features**, and an agent that
spends its first five iterations proving that again has learned nothing that was not already
published.

### The finding we would lead with

We ran one controlled experiment before the agent ever started, changing exactly one thing
from the official baseline — the loss — and found something we did not expect.

**With a pairwise ranking loss, the pair-sampling weight matters more than the loss function,
and it decides the sign of the result.**

| Same model, same loss, only the sampling changed | primary | vs baseline |
|---|---|---|
| Sample users **uniformly** | 0.5982 | −0.0034 |
| Sample users **weighted by positive count** | 0.6032 | +0.0016 |

A swing of 0.005 — larger than most model changes on this dataset — from a detail most people
would not think to mention. The reason is that GAUC is not a uniform average: it averages
per-user AUC **weighted by positive count**, over only those users with
`0 < positives < impressions`. Uniform pair sampling therefore optimises a genuinely different
quantity from the one being scored.

Why this matters for an *agent*: "switch to a ranking loss" is underspecified as an
instruction. An agent that implements the obvious thing measures −0.0034, concludes the
single most promising direction is refuted, and abandons the entire tier on the strength of an
implementation detail. So the idea bank does not say "use BPR" — it says which way to sample
the pairs and what happens if you do not. Encoding *how an idea fails* turned out to be worth
more than encoding the idea.

(We got this wrong ourselves on the first pass: the original experiment sampled users
uniformly and carried a comment claiming that matched GAUC. It does not. That is why the
control exists.)

### The rule we made mechanical instead of promising it

KuaiRand-Pure is a public dataset and the test labels are physically on our disk. "Hidden
test" means the organisers score their own copy — not that we are unable to read ours. An LLM
writing its own pipeline code will find those labels if nothing stops it, and tuning on test
is the easiest possible way for an agent to look brilliant and score badly.

So we did not put this in a prompt and hope. `evaluate.score(..., "test")` refuses to run
unless `ALLOW_TEST_SCORING=1` is set explicitly; scored iterations only ever execute
`--split val`; and on any non-KuaiRand task the materialised `test.csv` is written with the
target column removed, so no-peeking is enforced by absence rather than by instruction. The
system prompt states the rule too, but it is the weakest of the four guarantees, not the
only one.

## How failures are handled

Robustness is not measured by how rarely the agent fails — it is measured by what happens
next. The design assumption is that generated code fails often, so nothing in the loop is
allowed to treat a failure as exceptional.

```
pipeline fails
   │
   ├─ sandbox classifies it: syntax │ import │ data │ runtime │ oom │ timeout │ contract │ eval
   │
   ├─ journal the error (class + a 1500-char excerpt, never a whole traceback)
   │
   ├─ attempts < 3 ──► schedule a debug node
   │                   the repair prompt gets: the error class, the excerpt, the last stdout,
   │                   the failing code, AND what previous attempts already tried
   │                   ("do not repeat it — something about the diagnosis was wrong")
   │
   └─ attempts = 3 ──► node.status = "dead"; journal a recovery event; the policy routes
                       around it and the run continues from elsewhere
```

Details that carry the weight:

- **A failed node never stops the run.** Three repairs, then dead, then route around. The
  policy's *highest* priority is a repairable leaf, because a broken program is the cheapest
  thing in the tree to make valuable.
- **The repair prompt is scope-locked.** "Fix the failure. Change nothing else." A repair that
  also improves the model makes it impossible to tell whether the fix worked.
- **A malformed submission is a repairable error, not a crash.** `validate()` mirrors the
  official checker's checks, in the same order, and returns `(ok, message)` — so a wrong
  header or a misaligned row count comes back as `error_class="contract"` and gets three
  attempts like anything else.
- **Timeouts kill the whole process group**, not just the child, so a run cannot accumulate
  orphaned trainers. CI asserts that no pipeline process outlives the test run.
- **The API itself is a failure mode.** LLM calls retry with exponential backoff and jitter,
  and the client abstracts two providers behind one interface — so a mid-run rate limit is a
  configuration change, not a lost run.
- **Crash recovery.** State is checkpointed atomically every iteration; `--resume RUN_ID`
  restarts from it. `RESULTS.md` regenerates from the journal alone, so a run that dies at
  hour four still has a current deliverable.
- **Malformed journal lines are counted and skipped**, never raised. A reporting bug must
  never be the reason a finished run has no report.

Fault injection is a first-class test suite: every error class above has a fixture pipeline
that produces it, and the tests assert the classification, the repair routing and the
route-around.

**From the scored run:**

| | |
|---|---|
| Failed steps | `TODO` |
| Recovered automatically | `TODO` |
| Nodes abandoned after 3 repairs and routed around | `TODO` |
| Error classes seen | `TODO: from RESULTS.md's error-class table` |
| **Manual interventions** | `TODO` — target is 0; every human touch is logged in `interventions.md` and treated as a bug in the agent |

## What it cost

Resource use is a scored criterion, and it is only scored among submissions that beat the
baseline — so the design goal is *cheap given that it works*, never *cheap*.

Where the token budget goes, by construction:

- **One LLM call per node.** No planner/coder/critic personas. Multiple personas burn scored
  tokens to buy an effect we could not measure.
- **The static block is cached.** System prompt, task card, data card and library whitelist
  are byte-identical on every call in a run and marked for prompt caching, so from the second
  call they bill as a cache read.
- **The agent never sees the data.** It reads a generated markdown data card, capped, instead
  of any CSV.
- **History is compact.** Past *hypotheses* and metric deltas travel into the prompt; past
  *code* never does. Only the parent's code is sent.
- **Errors are excerpts.** A 1500-character slice of the traceback, not the traceback.
- **No GPU.** 100 iterations of the official baseline is roughly 28 minutes of one CPU core.
  Wall-clock is scored; a GPU would not have helped.

**From the scored run:**

| | |
|---|---|
| Iterations used (of 50) | `TODO` |
| Wall-clock to convergence | `TODO` |
| Total LLM tokens (input + output) | `TODO` |
| — input (incl. cache reads) | `TODO` |
| — output | `TODO` |
| Cache read fraction of input | `TODO` |
| GPU-hours | 0 — CPU only, by design |
| Model | `TODO: from RESULTS.md` |

`TODO: all of the above come straight out of runs/<run_id>/RESULTS.md, which is generated
from the journal. Do not retype them by hand.`

## Try it in three commands, with no API key

The whole loop runs offline against stubbed seams — which is also how CI runs it on every
push, so it cannot rot:

```bash
pip install -r requirements.txt
python -m orchestrator.run --task kuairand-pure --mode smoke
python -m orchestrator.report runs/<run_id>
```

And to prove it is not KuaiRand-shaped, a completely different problem — continuous target,
no groups, RMSE, a different submission schema — from a synthetic fixture:

```bash
python tools/make_demo_data.py
python -m orchestrator.run --task demo-regression --mode smoke
```

## Built with

Two dependency sets, kept deliberately apart. `requirements.txt` is what the **orchestrator**
needs; `requirements-pipeline.txt` is the whitelist the **agent-written pipeline** may import
inside the sandbox. An import outside that whitelist is an `ErrorClass: import` the agent has
to repair, which is why the two lists are separate files rather than one.

### Development tools

- **VS Code** — primary editor
- **Claude Code** (Anthropic) — used as a pair-programming agent on the harness itself
- **git / GitHub** — `main` is the submitted tree; every fix landed as its own commit
- **GNU Make** — `make check` (lint + tests + a stubbed end-to-end run) is the merge gate
- **pytest 8.3.4** — 425 tests, including a fault-injection suite that SIGKILLs real
  subprocesses and asserts none of them survive
- **ruff 0.9.6** — lint, run over `orchestrator/`, `tests/` and `tools/`
- **GitHub Actions** — Ubuntu, Python 3.11, running lint, the full test suite and a stubbed
  end-to-end run on every push, with no API key and no dataset

No notebooks. Every result in this writeup comes from a scripted run that regenerates from
its own run directory, because a number produced in a notebook cell cannot be reproduced by
a judge.

### APIs used

- **OpenAI API** (`openai==1.66.5`) — the models actually driving runs: **gpt-5.6-terra**,
  gpt-5.1, gpt-4o. Newer reasoning models refuse function tools on `/v1/chat/completions`,
  so the adapter moves those models to `/v1/responses` rather than disabling reasoning.
- **Anthropic API** (`anthropic==0.51.0`) — supported as an alternate primary; the internal
  interface is the Anthropic Messages shape and OpenAI is adapted onto it.
- **Google Gemini API** — configurable fallback, over stdlib `urllib` rather than a third SDK.

Failover is deliberately narrow: auth or model-not-found disables a provider for the run,
429/5xx fails over for one call, and it **never** fails over on a malformed proposal — that
is the repair loop's job, and conflating the two would hide model-quality problems behind a
provider switch. `summary.json` records which provider actually served every call.

### Libraries and frameworks

**Orchestrator:** pandas 2.2.3 · pyyaml 6.0.2 · matplotlib 3.10.0 (trajectory plot) ·
pytest · ruff · the openai and anthropic SDKs.

**Pipeline sandbox whitelist:** numpy 2.4.1 · scipy 1.17.0 (sparse matrices for wide linear
and FM-style models) · pandas 2.3.3 · scikit-learn 1.8.0 (splitters, encoders, linear models,
calibration) · **LightGBM 4.7.0** (GBDT, and `lambdarank` for listwise ranking) ·
PyTorch 2.13.0.

Versions are pinned and the prompt states what those major versions *removed*
(`DataFrame.append`, `verbose_eval`, `early_stopping_rounds`), because six of eight
iterations in our first live run died on exactly those APIs.

### Datasets and assets

- **KuaiRand-Pure** (Kuaishou), the required benchmark — 1,141,112 train / 124,909
  validation / 170,588 test impressions, split by the organisers' date ranges. Used
  unmodified; row counts verified against the published figures.
- **The organisers' starter kit**, vendored under `vendor/starter_kit/` — its `baseline.py`
  is run untouched to reproduce the official FM baseline, and its Chinese-language README
  supplied the measured dead ends encoded in `orchestrator/ideas.yaml`.
- **A synthetic rent-prediction fixture** (`tools/make_demo_data.py`) — continuous target,
  no groups, RMSE, a different submission schema. Exists only to prove the orchestrator is
  not KuaiRand-shaped.

**No external training data of any kind.** That is the one disqualifying rule, and it is
enforced rather than promised: generated pipelines run with outbound sockets blocked, so a
pipeline that tried to download anything raises `NetworkBlocked` instead of succeeding
quietly.

## What we would do next

`TODO: write this after the run, from what the trajectory actually showed. Candidates, to be
kept only if the journal supports them:`

- `TODO: whether the search converged or ran out of budget, and what that implies about
  where the next iterations should go.`
- `TODO: which tier the agent got to. Reaching T2/T3 at all would be the interesting result;
  never reaching them is also a finding worth reporting honestly.`
- `TODO: the strongest untried idea remaining in the bank at the moment the run stopped —
  knowledge.retrieve() can answer this exactly.`
