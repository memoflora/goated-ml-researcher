"""The hands that write pipeline code: LLM calls, prompt assembly, proposal
parsing, and the repair loop.

OWNER: B (ML Engineer — Agent Runtime & Sandbox).

Two scored criteria live in this file. **Feasibility** is won by prompt discipline:
one call per node, a cached static block, the parent's code but never the history's
code, and an error *excerpt* rather than a traceback. **Robustness** is won by the
repair loop: up to three attempts, each seeing what the previous one already tried.

B owns the plumbing; **D owns the prompt text** in `orchestrator/prompts/*.md`. This
module defines the template variables and substitutes them. It does not author the
words — the fallbacks at the bottom exist only so the loop runs before D's files
land, and they say so out loud.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .contracts import Context, Node, Proposal

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000
MAX_REPAIR_ATTEMPTS = 3
MAX_RETRIES = 5
PROMPT_DIR = Path(__file__).parent / "prompts"

# Temperature per action. Never 0 for drafts: three identical drafts waste the
# draft phase, and varying only the seed does not vary the idea.
TEMPERATURE = {"draft": 1.0, "improve": 0.6, "repair": 0.3}

# The draft phase asks for three genuinely different starting points. The angle is
# a template variable D's draft.md renders; varying it is what makes the drafts differ.
DRAFT_ANGLES = (
    "Reproduce the official baseline exactly and correctly: a plain factorization "
    "machine over the five categorical fields. Correctness first, cleverness never.",
    "Reproduce the baseline's modelling choice but invest in the data path: careful "
    "encoding, unseen-value handling, and an honest train/validation separation.",
    "Reproduce the baseline, then spend the remaining effort on the training loop "
    "itself: epochs, learning rate, regularisation and early stopping on validation.",
)

_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")


class AgentError(RuntimeError):
    """The LLM could not be made to return a usable proposal."""


@dataclass
class Usage:
    """Per-call accounting. Reported for the Feasibility tier, so it must be exact."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    calls: int = 0

    @property
    def tokens_in(self) -> int:
        """Everything billed as input, including cache writes and reads."""
        return (self.input_tokens + self.cache_creation_input_tokens
                + self.cache_read_input_tokens)

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.calls += other.calls

    def as_dict(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_creation_input_tokens": self.cache_creation_input_tokens,
                "cache_read_input_tokens": self.cache_read_input_tokens,
                "tokens_in": self.tokens_in, "tokens_out": self.output_tokens,
                "calls": self.calls}


# The Proposal is forced through tool use rather than parsed out of prose: a fenced
# code block in a chat reply is a parsing bug waiting to happen at hour four.
PROPOSAL_TOOL = {
    "name": "submit_pipeline",
    "description": (
        "Submit one complete, self-contained pipeline.py together with the reasoning "
        "behind it. This is the only way to return work; never reply in prose."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hypothesis": {
                "type": "string",
                "description": (
                    "One paragraph stating WHY this change should raise validation "
                    "primary on KuaiRand-Pure, before saying what it is. Name the "
                    "property of the data or the model you are exploiting. This is "
                    "read by judges: it must be a claim that could turn out false."
                ),
            },
            "plan": {
                "type": "array", "items": {"type": "string"},
                "description": "3-6 bullets: WHAT changes, concretely.",
            },
            "code": {
                "type": "string",
                "description": (
                    "The complete new pipeline.py, top to bottom. Not a diff, not a "
                    "fragment, no placeholders or TODOs. It must run as written."
                ),
            },
            "idea_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "Ids of the ideas drawn on, from the supplied list. May be empty.",
            },
        },
        "required": ["hypothesis", "plan", "code"],
    },
}


# --------------------------------------------------------------------------- #
# the three seams A calls
# --------------------------------------------------------------------------- #

class Agent:
    """Turns a Context into a Proposal. One LLM call per node, no personas."""

    def __init__(self, client=None, *, model: str | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 on_usage: Callable[[str, Usage], None] | None = None,
                 on_recovery: Callable[[dict], None] | None = None,
                 prompt_dir: Path | None = None):
        self.client = client if client is not None else make_client()
        self.model = model or default_model_for(self.client)
        self.max_tokens = max_tokens
        self.on_usage = on_usage
        self.on_recovery = on_recovery            # API retries: a recovery, not an intervention
        self.prompt_dir = prompt_dir or PROMPT_DIR
        self.total = Usage()

    # -- public API ---------------------------------------------------------

    def draft(self, ctx: Context) -> Proposal:
        angle = ctx.draft_angle or DRAFT_ANGLES[ctx.iteration % len(DRAFT_ANGLES)]
        return self._call("draft", ctx, {"draft_angle": angle})

    def improve(self, ctx: Context, parent: Node) -> Proposal:
        if not ctx.parent_code:
            raise AgentError("improve() needs the parent's code in Context.parent_code")
        return self._call("improve", ctx, {
            "parent_code": ctx.parent_code,
            "parent_metrics": _fmt_metrics(ctx.parent_metrics),
            "parent_node_id": parent.id,
        })

    def repair(self, ctx: Context, node: Node) -> Proposal:
        """Fix a failed node. Each attempt sees the error *and* what the previous
        attempts already tried, or the model returns the same broken code three
        times and we burn the node for nothing."""
        result = node.exec_result
        if result is None:
            raise AgentError(f"repair() needs node {node.id} to have an exec_result")
        code = ctx.parent_code or _read_pipeline(node)
        return self._call("repair", ctx, {
            "parent_code": code,
            "error_class": result.error_class or "unknown",
            "error_excerpt": result.error_excerpt or "(no excerpt captured)",
            "stdout_tail": _clip(result.stdout_tail, 800),
            "attempt": node.repair_attempts + 1,
            "max_attempts": MAX_REPAIR_ATTEMPTS,
            "previous_attempts": _fmt_attempts(node),
        })

    # -- one call -----------------------------------------------------------

    def _call(self, kind: str, ctx: Context, variables: dict) -> Proposal:
        system = self._system_blocks(ctx)
        user = self._render(kind, ctx, variables)
        # Scan everything that leaves the process, not just the user turn: the data
        # card and the whitelist ride in the system block.
        _assert_no_secret("\n".join([b["text"] for b in system] + [user]))

        payload, usage = self._invoke(system, user, TEMPERATURE[kind])
        if not str(payload.get("hypothesis", "")).strip():
            # The Innovation criterion is scored on this field. One retry, then fail
            # loudly so A can mark the node rather than journalling an empty string.
            payload, retry_usage = self._invoke(
                system, user + "\n\n" + _HYPOTHESIS_NAG, TEMPERATURE[kind])
            usage.add(retry_usage)
            self._recover("empty_hypothesis", f"{kind} returned no hypothesis; retried once")
        if not str(payload.get("hypothesis", "")).strip():
            raise AgentError(f"{kind}: model returned an empty hypothesis twice")
        if not str(payload.get("code", "")).strip():
            raise AgentError(f"{kind}: model returned no code")

        self.total.add(usage)
        if self.on_usage:
            self.on_usage(kind, usage)
        return Proposal(
            hypothesis=payload["hypothesis"].strip(),
            plan=[str(b).strip() for b in payload.get("plan") or [] if str(b).strip()],
            code=_strip_fences(payload["code"]),
            idea_ids=[str(i) for i in payload.get("idea_ids") or []],
            tokens_in=usage.tokens_in, tokens_out=usage.output_tokens, model=self.model,
        )

    def _invoke(self, system: list[dict], user: str, temperature: float
                ) -> tuple[dict, Usage]:
        """One request, with backoff on 429/5xx. A retried call costs tokens but
        never costs an iteration."""
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                message = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    temperature=temperature, system=system,
                    messages=[{"role": "user", "content": user}],
                    tools=[PROPOSAL_TOOL],
                    tool_choice={"type": "tool", "name": PROPOSAL_TOOL["name"]},
                )
            except Exception as exc:                      # noqa: BLE001 - classified below
                if not _retryable(exc) or attempt == MAX_RETRIES - 1:
                    raise AgentError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc
                delay = _backoff(attempt)
                self._recover("api_retry",
                              f"{type(exc).__name__} on attempt {attempt + 1}; "
                              f"retrying in {delay:.1f}s")
                time.sleep(delay)
                last = exc
                continue

            usage = _usage_of(message)
            payload = _tool_input(message, PROPOSAL_TOOL["name"])
            if payload is None:
                if attempt == MAX_RETRIES - 1:
                    raise AgentError("model never returned a submit_pipeline tool call")
                self._recover("no_tool_use", "model replied in prose; retrying")
                self.total.add(usage)          # a wasted call still costs tokens
                continue
            return payload, usage
        raise AgentError(f"LLM call failed after {MAX_RETRIES} attempts: {last}")

    def _recover(self, kind: str, detail: str) -> None:
        if self.on_recovery:
            self.on_recovery({"event": "recovery", "recovery": kind, "detail": detail})

    # -- prompt assembly ----------------------------------------------------

    def _system_blocks(self, ctx: Context) -> list[dict]:
        """The static block, cached. Identical on every call in a run, so from the
        second call onward it is billed as a cache read instead of input."""
        static = "\n\n".join([
            self._prompt_text("system"),
            _task_card(ctx),
            "## Data card\n\n" + (ctx.data_card or "(data card not available)"),
            _whitelist_block(ctx),
        ])
        return [{"type": "text", "text": static,
                 "cache_control": {"type": "ephemeral"}}]

    def _render(self, kind: str, ctx: Context, variables: dict) -> str:
        values = {
            "run_id": ctx.run_id,
            "iteration": ctx.iteration,
            "budget": _fmt_budget(ctx),
            "ideas": _fmt_ideas(ctx),
            "history": _fmt_history(ctx),
            "draft_angle": "",
            "parent_code": "",
            "parent_metrics": "",
            "parent_node_id": "",
            "error_class": "",
            "error_excerpt": "",
            "stdout_tail": "",
            "attempt": "",
            "max_attempts": MAX_REPAIR_ATTEMPTS,
            "previous_attempts": "",
            **variables,
        }
        return Template(self._prompt_text(kind)).safe_substitute(values)

    def _prompt_text(self, name: str) -> str:
        path = self.prompt_dir / f"{name}.md"
        if path.is_file():
            return path.read_text()
        return _FALLBACK_PROMPTS[name]

    def prompt_size(self, kind: str, ctx: Context, variables: dict | None = None) -> int:
        """Rough prompt size in characters, for the budget log. A cap that is never
        measured is a cap that is never enforced."""
        system = sum(len(b["text"]) for b in self._system_blocks(ctx))
        return system + len(self._render(kind, ctx, variables or {}))


# --------------------------------------------------------------------------- #
# the repair loop
# --------------------------------------------------------------------------- #

def repair_exhausted(node: Node) -> bool:
    """A node gets three attempts, then A marks it dead and routes around it."""
    return node.repair_attempts >= MAX_REPAIR_ATTEMPTS


# --------------------------------------------------------------------------- #
# clients
# --------------------------------------------------------------------------- #

def load_dotenv(path: Path | None = None) -> None:
    """Read `.env` into the environment without overriding what is already set.

    Fifteen lines instead of a dependency. `.env` is gitignored, and `sandbox.py`
    strips every key-shaped variable out of the child environment, so a generated
    pipeline never sees one.
    """
    path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if sep and name and name not in os.environ:
            os.environ[name] = value.strip().strip('"').strip("'")


# A key pasted under the wrong variable is a 401 four hours into a run, or worse a
# key sent to the wrong vendor. Both are cheap to catch here.
_KEY_SHAPES = {"anthropic": ("sk-ant-",), "openai": ("sk-proj-", "sk-")}


def _check_key_shape(provider: str, key: str) -> None:
    if provider == "anthropic" and key.startswith("sk-proj-"):
        raise AgentError(
            "ANTHROPIC_API_KEY holds what looks like an OpenAI key (sk-proj-...). "
            "Set OPENAI_API_KEY instead, or TECHJAM_LLM=openai.")
    if provider == "openai" and key.startswith("sk-ant-"):
        raise AgentError(
            "OPENAI_API_KEY holds what looks like an Anthropic key (sk-ant-...). "
            "Set ANTHROPIC_API_KEY instead, or TECHJAM_LLM=anthropic.")


def make_client(*, api_key: str | None = None, provider: str | None = None):
    """Return an LLM client. Provider order: explicit argument, then TECHJAM_LLM,
    then whichever key is in the environment.

    The rest of this module speaks one shape — Anthropic's Messages API — and
    `OpenAIClient` adapts to it. Nothing above this line knows which provider is
    live, which is also our fallback if one of them starts rate-limiting us
    halfway through the scored run.
    """
    load_dotenv()
    choice = (provider or os.environ.get("TECHJAM_LLM") or "").lower()
    if choice == "stub":
        return StubClient()
    if not choice:
        if os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            choice = "openai"
        else:
            raise AgentError(
                "No LLM key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or set "
                "TECHJAM_LLM=stub to run against the stub client.")

    if choice == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise AgentError("TECHJAM_LLM=anthropic but ANTHROPIC_API_KEY is not set.")
        _check_key_shape("anthropic", key)
        import anthropic
        return anthropic.Anthropic(api_key=key, max_retries=0)  # we own the retry policy
    if choice == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise AgentError("TECHJAM_LLM=openai but OPENAI_API_KEY is not set.")
        _check_key_shape("openai", key)
        import openai
        return OpenAIClient(openai.OpenAI(api_key=key, max_retries=0))
    raise AgentError(f"unknown provider {choice!r}; expected anthropic, openai or stub")


def default_model_for(client) -> str:
    """The model id to use when none was configured."""
    if isinstance(client, StubClient):
        return "stub"
    if isinstance(client, OpenAIClient):
        load_dotenv()
        model = os.environ.get("TECHJAM_MODEL")
        if not model:
            raise AgentError(
                "Using the OpenAI provider, so TECHJAM_MODEL must name the model "
                "explicitly — I will not guess a model id and discover it is wrong "
                "four hours into a scored run. List what your key can reach with:\n"
                "  python -c \"import openai; "
                "print(sorted(m.id for m in openai.OpenAI().models.list()))\"")
        return model
    return os.environ.get("TECHJAM_MODEL", DEFAULT_MODEL)


class OpenAIClient:
    """Presents OpenAI chat completions in the Anthropic Messages shape.

    Two differences are load-bearing rather than cosmetic:

    * **Token accounting.** Anthropic reports `input_tokens` *excluding* cache reads,
      so the billed input is the sum of three fields. OpenAI's `prompt_tokens`
      *includes* `cached_tokens`. Summing the same way on both would double-count
      every cached token and overstate a scored deliverable, so we subtract here and
      `Usage.tokens_in` stays exact for both providers.
    * **Prompt caching.** Anthropic caches where we put a `cache_control` marker.
      OpenAI caches automatically on the longest matching prefix above a length
      threshold, with nothing to declare. Our static block is already first and byte
      identical across calls, so it benefits either way; the marker is simply dropped.
    """

    def __init__(self, client):
        self._client = client

    @property
    def messages(self):
        return self

    def create(self, *, model, max_tokens, temperature, system, messages,
               tools, tool_choice, **_ignored):
        completion = self._client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": _flatten_system(system)}, *messages],
            tools=[_as_openai_tool(t) for t in tools],
            tool_choice={"type": "function",
                         "function": {"name": tool_choice["name"]}},
        )
        return _AdaptedMessage(
            content=_as_tool_use_blocks(completion),
            usage=_as_anthropic_usage(completion.usage),
        )


@dataclass
class _ToolUseBlock:
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _AdaptedMessage:
    content: list
    usage: Usage


def _flatten_system(system) -> str:
    """The cached static block, as a plain string. `cache_control` is Anthropic-only."""
    if isinstance(system, str):
        return system
    return "\n\n".join(block["text"] for block in system)


def _as_openai_tool(tool: dict) -> dict:
    return {"type": "function",
            "function": {"name": tool["name"], "description": tool["description"],
                         "parameters": tool["input_schema"]}}


def _as_tool_use_blocks(completion) -> list:
    """Returns [] when the model replied in prose, which the caller retries."""
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return []
    for call in getattr(choices[0].message, "tool_calls", None) or []:
        try:
            payload = json.loads(call.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(payload, dict):
            return [_ToolUseBlock(name=call.function.name, input=payload)]
    return []


def _as_anthropic_usage(usage) -> Usage:
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0)
    return Usage(
        input_tokens=prompt - cached,        # OpenAI counts cached inside prompt_tokens
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cache_creation_input_tokens=0,       # automatic caching has no write cost
        cache_read_input_tokens=cached,
        calls=1,
    )


@dataclass
class _StubMessage:
    content: list
    usage: object


class StubClient:
    """Deterministic offline stand-in for the Anthropic client.

    Returns a pipeline that satisfies the contract, so `make check` exercises the
    whole loop — assembly, tool parsing, accounting, sandbox, scoring — in seconds
    with no key and no spend. Fault fixtures can be forced via `TECHJAM_STUB_FAULT`.
    """

    def __init__(self, fault: str | None = None):
        self.fault = fault or os.environ.get("TECHJAM_STUB_FAULT") or None
        self.calls: list[dict] = []

    def __getattr__(self, name):                     # client.messages.create(...)
        if name == "messages":
            return self
        raise AttributeError(name)

    def create(self, **kwargs) -> _StubMessage:
        self.calls.append(kwargs)
        prompt = json.dumps(kwargs.get("messages", []))
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        payload = {
            "hypothesis": (
                "Stubbed proposal. Impressions are dominated by a handful of very "
                "active users, so a popularity prior ranked within each user should "
                "beat random ordering while the real agent is offline."),
            "plan": ["load the split", "score by item frequency", "write submission.csv"],
            "code": _STUB_PIPELINE.replace("__TAG__", digest),
            "idea_ids": ["T0.item-popularity"],
        }
        block = type("ToolUse", (), {"type": "tool_use", "name": PROPOSAL_TOOL["name"],
                                     "input": payload})()
        usage = type("Usage", (), {"input_tokens": 1200, "output_tokens": 400,
                                   "cache_creation_input_tokens": 0,
                                   "cache_read_input_tokens": 0})()
        return _StubMessage(content=[block], usage=usage)


_STUB_PIPELINE = '''\
"""Stub pipeline (__TAG__) — item popularity within user. Not the agent's work."""
import argparse, collections, csv, json, os, time

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--split", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--subsample", type=float, default=None)
a = p.parse_args()

t0 = time.time()
rows = []
path = os.path.join(a.data_dir, "%s.csv" % a.split)
if os.path.isfile(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
counts = collections.Counter(r.get("video_id") for r in rows)
with open(os.path.join(a.out_dir, "submission.csv"), "w") as fh:
    fh.write("row_id,user_id,video_id,score\\n")
    for i, r in enumerate(rows):
        fh.write("%d,%s,%s,%f\\n" % (i, r.get("user_id", "u"), r.get("video_id", "v"),
                                     counts[r.get("video_id")]))
print("RESULT_JSON " + json.dumps(
    {"n_rows": len(rows), "train_seconds": round(time.time() - t0, 3),
     "notes": "stub item-popularity"}))
'''


# --------------------------------------------------------------------------- #
# formatting helpers — where token cost is won or lost
# --------------------------------------------------------------------------- #

def _task_card(ctx: Context) -> str:
    t = ctx.task
    return (
        f"## Task\n\n"
        f"Benchmark `{t.name}`. Rank each user's logged impressions by predicted "
        f"`long_view`. Scored on {' and '.join(t.metrics)}; primary is their mean.\n\n"
        f"- Official baseline (validation): {_fmt_metrics(t.baseline_val)}\n"
        f"- Attainable ceiling: primary {t.ceiling:.4f} — not 1.0. Roughly a quarter "
        f"of users have no positive label, so their nDCG is 0 for any model.\n"
        f"- Development uses train and validation only. The test split is hidden.\n"
    )


def _whitelist_block(ctx: Context) -> str:
    if not ctx.library_whitelist:
        return ("## Libraries\n\nThe standard library and numpy. Importing anything "
                "else fails the run.")
    return ("## Libraries\n\nImport only from this list; anything else fails the run:\n"
            + "\n".join(f"- {lib}" for lib in ctx.library_whitelist))


def _fmt_metrics(metrics: dict[str, float] | None) -> str:
    if not metrics:
        return "not scored"
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def _fmt_budget(ctx: Context) -> str:
    b = ctx.budget
    if b is None:
        return "not reported"
    return (f"{b.iters_left} iterations, {b.seconds_left / 60:.0f} minutes and "
            f"{b.tokens_left:,} tokens left")


def _fmt_ideas(ctx: Context) -> str:
    if not ctx.ideas:
        return "(no ideas supplied — propose your own, and say why)"
    out = []
    for idea in ctx.ideas:
        cite = f" [{idea.citation}]" if idea.citation else ""
        out.append(f"- **{idea.id}** (T{idea.tier}, ~{idea.est_minutes} min){cite}: "
                   f"{idea.title} — {idea.summary}")
    return "\n".join(out)


def _fmt_history(ctx: Context) -> str:
    """Hypothesis and metric delta only. Never past code — that is the single
    biggest lever on the token bill."""
    if not ctx.history:
        return "(nothing tried yet)"
    out = []
    for h in ctx.history:
        if h.primary is not None:
            outcome = f"primary {h.primary:.4f}"
            if h.delta_primary is not None:
                outcome += f" ({h.delta_primary:+.4f} vs baseline)"
        else:
            outcome = f"failed ({h.error_class or h.status})"
        out.append(f"- i{h.iteration} [{h.kind}] {outcome} — {_clip(h.hypothesis, 220)}")
    return "\n".join(out)


def _fmt_attempts(node: Node) -> str:
    if not node.repair_attempts:
        return "(this is the first repair attempt)"
    return (f"{node.repair_attempts} previous repair attempt(s) failed. Do not repeat "
            f"them: change your diagnosis, not just the syntax.")


def _usage_of(message) -> Usage:
    u = getattr(message, "usage", None)
    return Usage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        calls=1,
    )


def _tool_input(message, name: str) -> dict | None:
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == name:
            data = getattr(block, "input", None)
            if isinstance(data, dict):
                return data
    return None


def _retryable(exc: Exception) -> bool:
    """429, 5xx, timeouts and connection drops. Everything else is our bug."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return type(exc).__name__ in {
        "RateLimitError", "InternalServerError", "APITimeoutError",
        "APIConnectionError", "APIStatusError", "OverloadedError",
    }


def _backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    return min(cap, base * (2 ** attempt)) * (0.5 + random.random() / 2)


def _strip_fences(code: str) -> str:
    """Models sometimes wrap the whole file in a fence even inside a tool call."""
    text = code.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.rstrip() + "\n"


def _read_pipeline(node: Node) -> str:
    path = Path(node.workspace) / "pipeline.py"
    return path.read_text() if path.is_file() else ""


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _assert_no_secret(text: str) -> None:
    """A key in a prompt would end up in the journal, which is a public deliverable."""
    if _SECRET_RE.search(text):
        raise AgentError("refusing to send a prompt containing something key-shaped")


_HYPOTHESIS_NAG = (
    "Your last response left `hypothesis` empty. It is required, and it is the field "
    "the judges read. State in one paragraph WHY your change should raise validation "
    "primary, naming the property of the data or model you are exploiting, before you "
    "describe what the change is."
)


# Placeholders only. D owns orchestrator/prompts/*.md; the moment those files exist
# they are used instead and these are never read again.
_FALLBACK_PROMPTS = {
    "system": (
        "PLACEHOLDER PROMPT (orchestrator/prompts/system.md is not present yet; D owns it).\n\n"
        "You are an autonomous ML research agent working on a recommender ranking task. "
        "You write one complete, self-contained `pipeline.py` per turn, called as:\n\n"
        "    python pipeline.py --data-dir DIR --out-dir DIR --split {val,test} --seed N "
        "[--subsample F]\n\n"
        "It must train on train only for `--split val` and on train+validation for "
        "`--split test`, write `<out-dir>/submission.csv` with header "
        "`row_id,user_id,video_id,score`, print exactly one line "
        "`RESULT_JSON {\"n_rows\": int, \"train_seconds\": float, \"notes\": str}`, exit "
        "non-zero on failure, and never read stdin. `--subsample F` samples users, not rows.\n\n"
        "Hard rules: never use any data outside the supplied directory; never compute or "
        "print a score yourself; never touch the test labels. Always state WHY before WHAT."
    ),
    "draft": (
        "PLACEHOLDER PROMPT (orchestrator/prompts/draft.md is not present yet; D owns it).\n\n"
        "Iteration $iteration of run $run_id. Budget: $budget.\n\n"
        "Write the first pipeline. Your angle for this draft:\n\n$draft_angle\n\n"
        "What has been tried so far:\n$history\n\nIdeas available:\n$ideas\n\n"
        "Submit it with the submit_pipeline tool."
    ),
    "improve": (
        "PLACEHOLDER PROMPT (orchestrator/prompts/improve.md is not present yet; D owns it).\n\n"
        "Iteration $iteration of run $run_id. Budget: $budget.\n\n"
        "Current best pipeline (node $parent_node_id) scored: $parent_metrics\n\n"
        "```python\n$parent_code\n```\n\n"
        "What has been tried:\n$history\n\nIdeas available:\n$ideas\n\n"
        "Make exactly ONE focused change and submit the complete new file. Multi-change "
        "proposals make the trajectory unreadable and attribution impossible."
    ),
    "repair": (
        "PLACEHOLDER PROMPT (orchestrator/prompts/repair.md is not present yet; D owns it).\n\n"
        "This pipeline failed. Repair attempt $attempt of $max_attempts.\n\n"
        "Error class: $error_class\n\n$error_excerpt\n\n"
        "Last stdout:\n$stdout_tail\n\n$previous_attempts\n\n"
        "```python\n$parent_code\n```\n\n"
        "Fix the failure and nothing else. Do not improve the model, do not refactor. "
        "Submit the complete corrected file."
    ),
}
