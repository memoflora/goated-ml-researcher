# "Built with" — Devpost tag field

Devpost's *Built with* field takes short tags. Paste these, comma-separated:

```
python, openai, openai-gpt-5, anthropic-claude, google-gemini, pytorch, lightgbm,
scikit-learn, numpy, scipy, pandas, matplotlib, pytest, ruff, github-actions,
kuairand, vs-code, claude-code, make, yaml, json
```

## The same list, with what each is actually for

Keep this for the body of the writeup — the tag field alone does not say which
dependencies the *agent* may use versus which the *harness* uses, and that boundary is
load-bearing: an import outside the pipeline whitelist is a classified error the agent
has to repair.

**Language** — Python 3.11

**APIs**
- **OpenAI** (`openai==1.66.5`) — the models that actually drove runs: **gpt-5.6-terra**,
  gpt-5.1, gpt-4o. Newer reasoning models refuse function tools on
  `/v1/chat/completions`, so the adapter moves them to `/v1/responses` rather than
  disabling reasoning.
- **Anthropic** (`anthropic==0.51.0`) — supported as an alternate primary. The internal
  interface *is* the Anthropic Messages shape; OpenAI is adapted onto it.
- **Google Gemini** — configurable fallback, over stdlib `urllib` rather than a third SDK.

**Orchestrator libraries** — pandas 2.2.3 · pyyaml 6.0.2 · matplotlib 3.10.0 (trajectory
plot) · pytest 8.3.4 · ruff 0.9.6

**Pipeline sandbox whitelist** — what the agent-written `pipeline.py` may import:
numpy 2.4.1 · scipy 1.17.0 · pandas 2.3.3 · scikit-learn 1.8.0 · **LightGBM 4.7.0**
(including `lambdarank` for listwise ranking) · PyTorch 2.13.0

**Development tools** — VS Code · **Claude Code** (used as a pair-programming agent on the
harness itself) · git/GitHub · GNU Make (`make check` is the merge gate) · GitHub Actions
(Ubuntu, Python 3.11, running lint, 425 tests and a stubbed end-to-end run on every push,
with no API key and no dataset)

**No notebooks.** Every number in the writeup comes from a scripted run that regenerates
from its own run directory, because a result produced in a notebook cell cannot be
reproduced by a judge.

**Data** — **KuaiRand-Pure** (Kuaishou): 1,141,112 train / 124,909 validation / 170,588
test impressions, used unmodified, row counts verified against the published figures. The
organisers' starter kit is vendored so its `baseline.py` runs untouched. A synthetic
rent-prediction fixture exists only to prove the orchestrator is not KuaiRand-shaped.

**No external training data of any kind** — the one disqualifying rule, enforced rather
than promised: generated pipelines run with outbound sockets blocked, so a pipeline that
tried to download anything raises `NetworkBlocked` instead of succeeding quietly.
