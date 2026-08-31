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

import ast
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
            # The system prompt is templated too: it states the submission contract, and
            # that contract is per task. Rendering it raw silently told every task to
            # write KuaiRand's header.
            Template(self._prompt_text("system")).safe_substitute(_task_values(ctx)),
            _task_card(ctx),
            "## Data card\n\n" + (ctx.data_card or "(data card not available)"),
            _whitelist_block(ctx),
        ])
        return [{"type": "text", "text": static,
                 "cache_control": {"type": "ephemeral"}}]

    def _render(self, kind: str, ctx: Context, variables: dict) -> str:
        values = {
            **_task_values(ctx),
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
            # Explicit utf-8: prompts carry em dashes, and Windows would decode cp1252.
            return path.read_text(encoding="utf-8")
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

def dotenv_candidates() -> list[Path]:
    """Where we look for `.env`, in order. The repo root is the documented spot;
    the package directory and the cwd are here because that is where people
    actually put it, and a silently unconfigured box is worse than a forgiving
    search."""
    root = Path(__file__).resolve().parent.parent
    return [root / ".env", root / "orchestrator" / ".env", Path.cwd() / ".env"]


def load_dotenv(path: Path | None = None) -> Path | None:
    """Read `.env` into the environment without overriding what is already set.
    Returns the file it used, or None. Never logs a value.

    Fifteen lines instead of a dependency. `.env` is gitignored, and `sandbox.py`
    strips every key-shaped variable out of the child environment, so a generated
    pipeline never sees one.
    """
    if path is not None:
        found = Path(path)
    else:
        found = next((c for c in dotenv_candidates() if c.is_file()), None)
    if found is None or not found.is_file():
        return None
    for raw in found.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if sep and name and name not in os.environ:
            os.environ[name] = value.strip().strip('"').strip("'")
    return found


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
        self._no_temperature: set[str] = set()

    @property
    def messages(self):
        return self

    def create(self, *, model, max_tokens, temperature, system, messages,
               tools, tool_choice, **_ignored):
        kwargs = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "system", "content": _flatten_system(system)}, *messages],
            "tools": [_as_openai_tool(t) for t in tools],
            "tool_choice": {"type": "function", "function": {"name": tool_choice["name"]}},
        }
        if model not in self._no_temperature:
            kwargs["temperature"] = temperature

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:                       # noqa: BLE001 - re-raised below
            if not _rejects_temperature(exc) or "temperature" not in kwargs:
                raise
            # Reasoning models accept only the default temperature. Learn it once,
            # per model, and carry on: a 400 is not retryable, so without this the
            # first improve() call of the run would kill its node outright.
            self._no_temperature.add(model)
            kwargs.pop("temperature")
            completion = self._client.chat.completions.create(**kwargs)

        return _AdaptedMessage(
            content=_as_tool_use_blocks(completion),
            usage=_as_anthropic_usage(completion.usage),
        )

    def fixed_temperature_models(self) -> set[str]:
        """Models found to reject a custom temperature. Worth journalling: it means
        draft diversity rests entirely on the prompt angle, which is where we wanted
        it anyway."""
        return set(self._no_temperature)


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


def _rejects_temperature(exc: Exception) -> bool:
    """A 400 specifically about the temperature value, not any other bad request."""
    if getattr(exc, "status_code", None) != 400:
        return False
    return "temperature" in str(exc).lower()


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
            "code": (_STUB_PIPELINE
                     .replace("__TAG__", digest)
                     .replace("__HEADER__", _sniff_submission_header(kwargs))),
            "idea_ids": ["T0.item-popularity"],
        }
        block = type("ToolUse", (), {"type": "tool_use", "name": PROPOSAL_TOOL["name"],
                                     "input": payload})()
        usage = type("Usage", (), {"input_tokens": 1200, "output_tokens": 400,
                                   "cache_creation_input_tokens": 0,
                                   "cache_read_input_tokens": 0})()
        return _StubMessage(content=[block], usage=usage)


#: The submission header, stated verbatim in the system prompt, is the one piece of the
#: task the stub client cannot invent. It used to hardcode KuaiRand's, which made every
#: other task's stub run fail its structural check before it was ever scored.
_HEADER_RE = re.compile(r"row_id(?:,[A-Za-z_]\w*)+")


def _sniff_submission_header(kwargs: dict) -> str:
    """Recover the task's submission header out of the prompt we were just handed.

    `StubClient` sees a rendered prompt and nothing else — no `Context`, no `TaskSpec` —
    but `prompts/system.md` states the header literally, so it is recoverable. Falls back
    to KuaiRand's, which is what the old stub assumed unconditionally.
    """
    texts = [b.get("text", "") for b in (kwargs.get("system") or []) if isinstance(b, dict)]
    for message in kwargs.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            texts.append(content)
    for text in texts:
        found = _HEADER_RE.search(text)
        if found and found.group(0).count(",") >= 2:
            return found.group(0)
    return "row_id,user_id,video_id,score"


# The stub's pipeline has to satisfy the *real* validator, not a friendly one. That means
# reading the actual evaluation split and emitting rows in its order: `row_id` is a
# position inside that split, and the id columns are checked against it line by line. The
# previous version invented ids from `row_id % 97`, which aligns with nothing, so a stubbed
# run reached "submission failed validation" on every node and never scored one.
_STUB_PIPELINE = '''\
"""Stub pipeline (__TAG__) — reads the real split, ranks by item popularity.

Not the agent's work: nothing is trained. The only thing this file takes seriously is
alignment, because a misaligned submission is rejected before any model can be judged.

Two directory layouts are recognised: the raw KuaiRand-Pure directory (a split is a date
filter over the two standard logs, in file order) and one-CSV-per-split, which is what the
orchestrator materialises for every other task.
"""
import argparse, collections, csv, hashlib, json, os, time

COLUMNS = "__HEADER__".split(",")
ID_COLUMNS = COLUMNS[1:-1]
LOGS = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
RANGES = {"train": (20220408, 20220421), "valid": (20220422, 20220428),
          "test": (20220429, 20220508)}


def jitter(values, seed):
    """Deterministic tie-break in [0, 1). `hash()` is salted per process; this is not."""
    blob = ("\\x1f".join(values) + "|" + str(seed)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "big") / 2.0 ** 64


def header_of(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh), []) or []


def table_path(data_dir, split):
    for name in ([split + ".csv"] + (["val.csv"] if split == "valid" else [])):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    return None


def read_logs(data_dir, split):
    lo, hi = RANGES[split]
    rows = []
    for name in LOGS:
        with open(os.path.join(data_dir, name), newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            head = next(reader)
            date_at = head.index("date")
            id_at = [head.index(c) for c in ID_COLUMNS]
            for rec in reader:
                if lo <= int(rec[date_at]) <= hi:
                    rows.append([rec[i] for i in id_at])
    return rows


def read_table(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [[str(rec[c]) for c in ID_COLUMNS] for rec in csv.DictReader(fh)]


def train_target_mean(data_dir):
    """The target is the column train.csv has and test.csv does not, by construction."""
    train, test = table_path(data_dir, "train"), table_path(data_dir, "test")
    if not train or not test:
        return None
    held_out = set(header_of(test))
    extra = [c for c in header_of(train) if c not in held_out]
    if not extra:
        return None
    column, total, n = extra[-1], 0.0, 0
    with open(train, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            try:
                total += float(rec[column])
            except (TypeError, ValueError):
                continue
            n += 1
    return total / n if n else None


p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--split", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--subsample", type=float, default=None)
a = p.parse_args()

t0 = time.time()
split = {"val": "valid"}.get(a.split, a.split)
os.makedirs(a.out_dir, exist_ok=True)

if all(os.path.isfile(os.path.join(a.data_dir, n)) for n in LOGS):
    rows, layout = read_logs(a.data_dir, split), "starter_kit"
else:
    path = table_path(a.data_dir, split)
    if path is None:
        raise FileNotFoundError("no %s split under %s" % (split, a.data_dir))
    rows, layout = read_table(path), "table"

if layout == "starter_kit" and len(ID_COLUMNS) >= 2:
    counts = collections.Counter(r[-1] for r in rows)
    preds = [counts[r[-1]] + jitter(r, a.seed) for r in rows]
    model = "item popularity"
else:
    mean = train_target_mean(a.data_dir)
    preds = [mean] * len(rows) if mean is not None else [jitter(r, a.seed) for r in rows]
    model = "train-set mean" if mean is not None else "hashed pseudo-score"

with open(os.path.join(a.out_dir, "submission.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\\n")
    w.writerow(COLUMNS)
    for i, (ids, pred) in enumerate(zip(rows, preds)):
        w.writerow([i] + ids + ["%.6f" % pred])

print("RESULT_JSON " + json.dumps(
    {"n_rows": len(rows), "train_seconds": round(time.time() - t0, 3),
     "notes": "stub %s, layout=%s, split=%s" % (model, layout, split)}))
'''


# --------------------------------------------------------------------------- #
# formatting helpers — where token cost is won or lost
# --------------------------------------------------------------------------- #

def _task_card(ctx: Context) -> str:
    """The standing description of what is being solved.

    Every fact here comes off the `TaskSpec`, which is built from `tasks/<name>.yaml`.
    It used to be a hardcoded paragraph about ranking `long_view`, which quietly made the
    agent unusable on any other problem — and, worse, would have described the wrong task
    while still looking like a working prompt.
    """
    t = ctx.task
    lines = [f"## Task\n\nBenchmark `{t.name}` ({getattr(t, 'kind', 'ranking')})."]

    description = (getattr(t, "description", "") or "").strip()
    if description:
        lines.append("\n" + description)

    lines.append(
        f"\nScored on {', '.join(f'`{m}`' for m in t.metrics)}. "
        f"`primary` is what the search maximises: {_primary_expr(t)}."
    )
    if t.baseline_val:
        lines.append(f"- Baseline to beat (validation): {_fmt_metrics(t.baseline_val)}")
    if getattr(t, "ceiling", None) is not None:
        lines.append(
            f"- Attainable ceiling: primary {t.ceiling:.4f} — not 1.0. "
            "Judge progress against that number."
        )
    if getattr(t, "seed_std", None):
        lines.append(
            f"- Run-to-run noise of a fixed pipeline is about {t.seed_std:.4f}. "
            "A smaller difference than that is not a result."
        )
    lines.append("- Development uses train and validation only. The test split is held out.")
    return "\n".join(lines) + "\n"


def _fmt_dead_ends(cfg) -> str:
    """The task's measured-false claims, for the cached system block.

    These used to be typed into `system.md` as KuaiRand prose, which meant a second task
    would have inherited another dataset's conclusions as if they were its own. They now
    come from that task's idea bank, so an empty bank simply yields nothing.
    """
    try:
        from orchestrator.knowledge import dead_ends
    except ImportError:
        return "(none recorded yet)"
    path = getattr(cfg, "ideas_path", None) if cfg is not None else None
    try:
        entries = dead_ends(str(path) if path else None)
    except Exception:  # noqa: BLE001 - a broken bank must not kill the run
        return "(none recorded yet)"
    if not entries:
        return "(none recorded yet)"
    return "\n\n".join(f"- **{d.claim}** {d.verdict}" for d in entries)


def _task_values(ctx: Context) -> dict:
    """Task-derived `$placeholders` available to every prompt, system.md included.

    These are what let one set of prompt files serve any task. D writes the words; the
    values come off the `TaskSpec`, which comes off `tasks/<name>.yaml`.
    """
    t = ctx.task
    cols = tuple(getattr(t, "submission_columns", ("row_id", "user_id", "video_id", "score")))
    pred = getattr(t, "prediction_column", "score")
    group = None
    cfg = getattr(t, "config", None)
    if cfg is not None:
        group = getattr(cfg.data, "group", None)

    if group:
        subsample_note = (
            f"sampling whole **{group}** groups, not rows. Row sampling silently breaks "
            f"the grouped metrics, which are computed within a {group}."
        )
        order_note = f"only the relative order within a `{group}` matters"
    else:
        subsample_note = "sampling **rows** uniformly at random with the given seed."
        order_note = "the value itself matters, not just its rank"

    return {
        "dead_ends": _fmt_dead_ends(cfg),
        "task_name": t.name,
        "task_kind": getattr(t, "kind", "ranking"),
        "task_description": (getattr(t, "description", "") or "").strip(),
        "submission_header": ",".join(cols),
        "prediction_column": pred,
        "group_column": group or "",
        "subsample_note": subsample_note,
        "order_note": order_note,
        "metric_names": ", ".join(t.metrics),
        "primary_expr": _primary_expr(t),
    }


def _primary_expr(t) -> str:
    """Write `primary` out as a signed expression so its direction is never ambiguous."""
    parts = getattr(t, "primary_parts", None) or tuple(t.metrics)
    try:
        from orchestrator import metrics as _M

        terms = [(p if _M.get(p).greater_is_better else f"-{p}") for p in parts]
    except (ImportError, KeyError):
        terms = list(parts)
    return terms[0] if len(terms) == 1 else "mean of " + ", ".join(terms)


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
            if h.delta_vs_baseline is not None:
                outcome += f" ({h.delta_vs_baseline:+.4f} vs baseline)"
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


# --------------------------------------------------------------------------- #
# ReplayAgent — the offline path
# --------------------------------------------------------------------------- #

#: Set to `replay` (or `scripted`) to make `get_agent()` serve canned pipelines.
REPLAY_ENV = "TECHJAM_AGENT"
#: Where the canned pipelines live. Override to replay a different trajectory.
REPLAY_DIR_ENV = "TECHJAM_REPLAY_DIR"
DEFAULT_REPLAY_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "replay"
#: Each canned pipeline carries this token where the task's submission header belongs.
REPLAY_HEADER_TOKEN = "__SUBMISSION_HEADER__"
DEFAULT_SUBMISSION_HEADER = "row_id,user_id,video_id,score"


def _replay_reasoning(code: str, name: str) -> tuple[str, list[str]]:
    """Lift the hypothesis and plan out of a canned pipeline's own docstring.

    The alternative is to caption the file from out here, which would put a sentence in
    the journal that nobody had to keep true as the file changed.
    """
    try:
        doc = (ast.get_docstring(ast.parse(code)) or "").strip()
    except SyntaxError:
        doc = ""
    paragraphs = [p.strip().replace("\n", " ") for p in doc.split("\n\n") if p.strip()]
    paragraphs = [p for p in paragraphs if not p.startswith("Served offline")]
    if not paragraphs:
        return f"Replayed {name}: a canned pipeline, served offline.", [f"run {name}"]
    return " ".join(paragraphs), [
        paragraphs[0],
        "read the evaluation split in row_id order",
        "write submission.csv and print RESULT_JSON",
    ]


class ReplayAgent:
    """Serves canned pipeline files in sequence instead of calling a model.

    Same seam as `Agent` — `draft(ctx)`, `improve(ctx, parent)`, `repair(ctx, node)` — so
    the orchestrator cannot tell the difference, and everything downstream of the LLM call
    runs for real: sandbox, validator, scorer, tree, journal, final submission. No key, no
    network, no spend, and the same trajectory every time, which is what makes it usable as
    a CI gate rather than only as a demo.

    Each call consumes the next file, wrapping round at the end. A repair consumes one too:
    offline there is no new information to repair *with*, so re-serving the file that just
    failed would spend the node's three attempts on the same failure and teach us nothing.

    The one thing substituted into each file is the task's submission header, which is the
    single fact a canned pipeline cannot know and cannot guess. Everything else it works
    out from the directory it is pointed at.
    """

    model = "replay-agent-v1"

    def __init__(
        self,
        pipelines: list[Path | str] | None = None,
        *,
        directory: Path | str | None = None,
    ) -> None:
        if pipelines is not None:
            self.paths = [Path(p) for p in pipelines]
        else:
            root = Path(directory or os.environ.get(REPLAY_DIR_ENV) or DEFAULT_REPLAY_DIR)
            self.paths = sorted(p for p in root.glob("*.py") if not p.name.startswith("_"))
            if not self.paths:
                raise AgentError(
                    f"no canned pipelines in {root}; set {REPLAY_DIR_ENV} to a directory "
                    "of *.py pipelines, or pass them explicitly"
                )
        self.cursor = 0
        self.calls: list[str] = []
        self.total = Usage()

    # -- seam ---------------------------------------------------------------

    def draft(self, ctx: Context) -> Proposal:
        return self._serve(ctx, "draft", "first program of the run")

    def improve(self, ctx: Context, parent: Node) -> Proposal:
        return self._serve(ctx, "improve", f"one step on from {parent.id}")

    def repair(self, ctx: Context, node: Node) -> Proposal:
        return self._serve(
            ctx, "repair", f"{node.id} failed with error_class={ctx.error_class}"
        )

    # -- internals ----------------------------------------------------------

    def _serve(self, ctx: Context, kind: str, note: str) -> Proposal:
        path = self.paths[self.cursor % len(self.paths)]
        self.cursor += 1
        self.calls.append(kind)

        columns = getattr(getattr(ctx, "task", None), "submission_columns", None)
        header = ",".join(columns) if columns else DEFAULT_SUBMISSION_HEADER
        code = path.read_text(encoding="utf-8").replace(REPLAY_HEADER_TOKEN, header)

        hypothesis, plan = _replay_reasoning(code, path.name)
        return Proposal(
            hypothesis=(
                f"{hypothesis}\n\n(Replayed offline from {path.name} — {note}. "
                "No model was called and no tokens were spent.)"
            ),
            plan=plan,
            code=code,
            idea_ids=[i.id for i in (ctx.ideas or [])[:1]],
            tokens_in=0,
            tokens_out=0,
            model=f"{self.model}:{path.stem}",
        )


# ---------------------------------------------------------------------------
# Module-level seam. OWNER: B — added by C to unblock the first real run.
#
# contracts.md §3 freezes this seam as module functions:
#     draft(ctx) -> Proposal / improve(ctx, parent) / repair(ctx, node)
# but they landed only as methods on `Agent`. A's run.py resolves each seam with
# hasattr(module, name), so the probe failed and every run silently fell back to
# StubAgent — completing, journalling and reporting success having never called
# the LLM. These restore the frozen shape; `Agent` itself is unchanged.
# ---------------------------------------------------------------------------

_DEFAULT: Agent | ReplayAgent | None = None


def replay_requested() -> bool:
    """Whether this process should run the canned agent instead of the real one."""
    return os.environ.get(REPLAY_ENV, "").strip().lower() in {"replay", "scripted"}


def get_agent(**kwargs) -> Agent | ReplayAgent:
    """The shared agent, built on first use.

    Never at import time: run.py imports this module merely to probe the seam,
    and constructing a client needs an API key a stubbed smoke run does not have.

    `TECHJAM_AGENT=replay` swaps in `ReplayAgent`, which is how the whole loop is
    exercised end to end with no key and no spend. It is honoured here rather than in
    run.py's `--agent` flag because this is the seam run.py resolves, so no other file
    has to change to get the offline path.
    """
    global _DEFAULT
    if _DEFAULT is None or kwargs:
        _DEFAULT = ReplayAgent() if replay_requested() else Agent(**kwargs)
    return _DEFAULT


def draft(ctx: Context) -> Proposal:
    return get_agent().draft(ctx)


def improve(ctx: Context, parent: Node) -> Proposal:
    return get_agent().improve(ctx, parent)


def repair(ctx: Context, node: Node) -> Proposal:
    return get_agent().repair(ctx, node)
