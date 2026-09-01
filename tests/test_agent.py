"""Tests for the agent runtime: prompt assembly, token accounting, structured
output, and the retry/repair behaviour that keeps a six-hour run alive."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from orchestrator import agent as agent_mod
from orchestrator.agent import Agent, AgentError, StubClient, Usage
from orchestrator.contracts import (
    PIPELINE_CLI,
    Budget,
    Context,
    ExecResult,
    HistoryEntry,
    Idea,
    Node,
    TaskSpec,
)

_needs_openai = pytest.mark.skipif(
    importlib.util.find_spec("openai") is None,
    reason="optional provider; `pip install openai` to run these",
)

# `make_client()` only reaches `import anthropic` once a key survives the shape check, so
# most of TestProviderSelection runs without the SDK. The one test that asserts on the
# constructed client cannot, and gating the class on `openai` alone made it fail on any
# box with openai installed and anthropic not — which is most CI images.
_needs_anthropic = pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="optional provider; `pip install anthropic` to run this",
)

TASK = TaskSpec(
    name="kuairand-pure", data_dir=Path("data"), metrics=("gauc", "ndcg@5"),
    baseline_val={"gauc": 0.6674, "ndcg@5": 0.5357, "primary": 0.6016},
    baseline_test={"gauc": 0.6610, "ndcg@5": 0.5282, "primary": 0.5946},
)

GOOD_PAYLOAD = {
    "hypothesis": "Active users dominate impressions, so a per-user prior should help.",
    "plan": ["add the prior", "retrain", "write submission"],
    "code": "print('hello')\n",
    "idea_ids": ["T1.user-ctr-prior"],
}


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #

class FakeUsage:
    def __init__(self, i=1000, o=500, cc=0, cr=0):
        self.input_tokens, self.output_tokens = i, o
        self.cache_creation_input_tokens, self.cache_read_input_tokens = cc, cr


class FakeBlock:
    def __init__(self, payload, name="submit_pipeline", type="tool_use"):
        self.type, self.name, self.input = type, name, payload


class FakeMessage:
    def __init__(self, content, usage):
        self.content, self.usage = content, usage


class FakeClient:
    """Scripted client. `script` is a list of payloads, exceptions, or FakeMessages."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else GOOD_PAYLOAD
        if isinstance(item, Exception):
            raise item
        if isinstance(item, FakeMessage):
            return item
        return FakeMessage([FakeBlock(item)], FakeUsage())


class Rate429(Exception):
    status_code = 429


class Bad400(Exception):
    status_code = 400


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(agent_mod.time, "sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def no_live_http(monkeypatch):
    """Nothing in this file may reach an API.

    `GeminiClient` takes its transport as an argument so the tests can hand it a
    recorder; this makes the real one explode if anything ever forgets. A suite that
    *could* bill us is a suite nobody trusts to run in a loop.
    """
    def refuse(*_args, **_kwargs):
        raise AssertionError("the test suite must never make a live API call")

    monkeypatch.setattr(agent_mod, "_gemini_transport", refuse)


@pytest.fixture(autouse=True)
def no_ambient_dotenv(monkeypatch):
    """Tests must not care whether the developer running them has a `.env`.

    Without this, provider-selection tests pass in CI (no key) and fail on a
    configured box (key found) — the worst kind of flake, because it disagrees
    with itself across machines. `load_dotenv(explicit_path)` still works, so
    TestDotenv exercises the real thing."""
    monkeypatch.setattr(agent_mod, "dotenv_candidates", list)


@pytest.fixture
def ctx():
    return Context(
        task=TASK, data_card="## Data\n1.1M train rows, long_view base rate 0.42.",
        ideas=[Idea(id="T1.user-ctr-prior", tier=1, title="Smoothed user CTR prior",
                    summary="Add a Bayesian-smoothed per-user CTR feature.",
                    citation="Chapelle 2014", est_minutes=10, prerequisites=[])],
        history=[HistoryEntry(iteration=1, node_id="n000", kind="draft", hypothesis="FM baseline",
                              status="ok", primary=0.6016, delta_vs_baseline=0.0)],
        budget=Budget(iters_left=42, seconds_left=18000, tokens_left=400_000),
        library_whitelist=("numpy", "scipy"), run_id="r-test", iteration=2,
        pipeline_cli=PIPELINE_CLI, baseline_val=TASK.baseline_val,
    )


def node(**kw):
    base = {"id": "n001", "parent_id": None, "kind": "draft", "iteration": 1,
            "workspace": Path("/tmp/nonexistent-node")}
    base.update(kw)
    return Node(**base)


#: Everything `make_client()` will auto-detect. A test about having no key has to clear
#: all of them, or it passes on CI and fails on the machine that actually has them.
KEY_ENVS = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "TECHJAM_LLM", "TECHJAM_FALLBACK_LLM", "TECHJAM_FALLBACK_MODEL",
)


def no_keys(monkeypatch):
    for var in KEY_ENVS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #

class TestPromptAssembly:
    def test_static_block_is_cached(self, ctx):
        a = Agent(FakeClient([]), on_usage=None)
        a.draft(ctx)
        system = a.client.calls[0]["system"]
        assert len(system) == 1, "one static block, or caching does not apply"
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_static_block_is_byte_identical_across_calls(self, ctx):
        """If it drifts, every call is a cache miss and the token bill doubles."""
        a = Agent(FakeClient([]))
        a.draft(ctx)
        ctx.iteration = 9
        ctx.parent_code = "print(1)\n"
        a.improve(ctx, node())
        first, second = (c["system"][0]["text"] for c in a.client.calls[:2])
        assert first == second

    def test_static_block_carries_task_data_card_and_whitelist(self, ctx):
        a = Agent(FakeClient([]))
        a.draft(ctx)
        text = a.client.calls[0]["system"][0]["text"]
        assert "kuairand-pure" in text
        assert "long_view base rate 0.42" in text          # C's data card
        assert "numpy" in text and "scipy" in text          # the whitelist
        assert "0.8645" in text                             # the real ceiling

    def test_history_carries_hypotheses_but_never_code(self, ctx):
        ctx.history.append(HistoryEntry(iteration=2, node_id="n001", kind="improve",
                                        hypothesis="Add a recency feature",
                                        status="ok", primary=0.61, delta_vs_baseline=0.0084))
        a = Agent(FakeClient([]))
        ctx.parent_code = "SECRET_MARKER_PARENT_CODE = 1\n"
        a.improve(ctx, node())
        user = a.client.calls[0]["messages"][0]["content"]
        assert "Add a recency feature" in user
        assert user.count("SECRET_MARKER_PARENT_CODE") == 1, \
            "only the parent's code may appear, never history code"

    def test_failed_history_entries_report_their_error_class(self, ctx):
        # Asserted against `improve`, not `draft`: D's draft.md is a cold start and
        # deliberately renders no $history, so this used to pass only against the
        # placeholder prompt and started failing the moment the real one landed.
        ctx.history = [HistoryEntry(iteration=1, node_id="n000", kind="draft", hypothesis="try FM",
                             primary=None, delta_vs_baseline=None,
                                    status="buggy", error_class="import")]
        ctx.parent_code = "x = 1\n"
        a = Agent(FakeClient([]))
        a.improve(ctx, node())
        assert "failed (import)" in a.client.calls[0]["messages"][0]["content"]

    def test_d_prompt_files_win_over_the_fallbacks(self, ctx, tmp_path):
        (tmp_path / "draft.md").write_text("REAL DRAFT PROMPT for $run_id, iter $iteration", encoding="utf-8")
        a = Agent(FakeClient([]), prompt_dir=tmp_path)
        a.draft(ctx)
        user = a.client.calls[0]["messages"][0]["content"]
        assert user == "REAL DRAFT PROMPT for r-test, iter 2"
        assert "PLACEHOLDER" not in user

    def test_unknown_template_variables_are_left_alone(self, ctx, tmp_path):
        """safe_substitute, so a typo in D's prompt cannot crash a run at hour four."""
        (tmp_path / "draft.md").write_text("$run_id and $not_a_variable", encoding="utf-8")
        a = Agent(FakeClient([]), prompt_dir=tmp_path)
        a.draft(ctx)
        assert a.client.calls[0]["messages"][0]["content"] == "r-test and $not_a_variable"

    def test_prompt_size_is_measurable(self, ctx):
        a = Agent(FakeClient([]))
        assert a.prompt_size("draft", ctx) > 100


class TestDraftPhase:
    def test_drafts_are_not_temperature_zero(self, ctx):
        a = Agent(FakeClient([]))
        a.draft(ctx)
        assert a.client.calls[0]["temperature"] > 0.5

    def test_successive_drafts_get_different_angles(self, ctx):
        a = Agent(FakeClient([]))
        seen = set()
        for i in range(3):
            ctx.iteration = i
            a.draft(ctx)
            seen.add(a.client.calls[-1]["messages"][0]["content"])
        assert len(seen) == 3, "three identical drafts waste the draft phase"

    def test_explicit_angle_overrides(self, ctx):
        ctx.draft_angle = "Try a pure popularity ranker first."
        a = Agent(FakeClient([]))
        a.draft(ctx)
        assert "pure popularity ranker" in a.client.calls[0]["messages"][0]["content"]


class TestStructuredOutput:
    def test_tool_use_is_forced(self, ctx):
        a = Agent(FakeClient([]))
        a.draft(ctx)
        call = a.client.calls[0]
        assert call["tool_choice"] == {"type": "tool", "name": "submit_pipeline"}
        assert call["tools"][0]["input_schema"]["required"] == ["hypothesis", "plan", "code"]

    def test_proposal_fields_come_through(self, ctx):
        p = Agent(FakeClient([])).draft(ctx)
        assert p.hypothesis.startswith("Active users dominate")
        assert p.plan == ["add the prior", "retrain", "write submission"]
        assert p.idea_ids == ["T1.user-ctr-prior"]
        assert p.model == agent_mod.DEFAULT_MODEL

    def test_prose_reply_is_retried(self, ctx):
        prose = FakeMessage([FakeBlock(None, type="text")], FakeUsage())
        recoveries = []
        a = Agent(FakeClient([prose, GOOD_PAYLOAD]), on_recovery=recoveries.append)
        assert a.draft(ctx).code == "print('hello')\n"
        assert recoveries[0]["recovery"] == "no_tool_use"

    def test_empty_hypothesis_is_retried_once_then_fails(self, ctx):
        empty = {**GOOD_PAYLOAD, "hypothesis": "  "}
        recoveries = []
        a = Agent(FakeClient([empty, GOOD_PAYLOAD]), on_recovery=recoveries.append)
        assert a.draft(ctx).hypothesis                       # recovered
        assert recoveries[0]["recovery"] == "empty_hypothesis"

        b = Agent(FakeClient([empty, empty]))
        with pytest.raises(AgentError, match="empty hypothesis twice"):
            b.draft(ctx)

    def test_the_nag_explains_why_the_field_matters(self, ctx):
        a = Agent(FakeClient([{**GOOD_PAYLOAD, "hypothesis": ""}, GOOD_PAYLOAD]))
        a.draft(ctx)
        assert "WHY" in a.client.calls[1]["messages"][0]["content"]

    def test_empty_code_is_an_error(self, ctx):
        a = Agent(FakeClient([{**GOOD_PAYLOAD, "code": "   "}]))
        with pytest.raises(AgentError, match="no code"):
            a.draft(ctx)

    def test_stray_code_fences_are_stripped(self, ctx):
        a = Agent(FakeClient([{**GOOD_PAYLOAD, "code": "```python\nprint(1)\n```"}]))
        assert a.draft(ctx).code == "print(1)\n"


class TestAccounting:
    def test_tokens_in_includes_cache_reads_and_writes(self, ctx):
        msg = FakeMessage([FakeBlock(GOOD_PAYLOAD)], FakeUsage(i=100, o=50, cc=900, cr=300))
        p = Agent(FakeClient([msg])).draft(ctx)
        assert p.tokens_in == 1300 and p.tokens_out == 50

    def test_totals_accumulate_and_are_reported(self, ctx):
        seen = []
        a = Agent(FakeClient([]), on_usage=lambda kind, u: seen.append((kind, u.tokens_in)))
        a.draft(ctx)
        ctx.parent_code = "print(1)\n"
        a.improve(ctx, node())
        assert a.total.calls == 2 and a.total.tokens_in == 2000
        assert [k for k, _ in seen] == ["draft", "improve"]

    def test_a_wasted_call_still_costs_tokens(self, ctx):
        """A prose reply we throw away was still billed. Under-reporting spend
        would misstate a scored deliverable."""
        prose = FakeMessage([FakeBlock(None, type="text")], FakeUsage(i=700, o=20))
        a = Agent(FakeClient([prose, GOOD_PAYLOAD]))
        a.draft(ctx)
        assert a.total.calls == 2 and a.total.tokens_in == 1700

    def test_usage_dict_shape(self):
        u = Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=2, calls=1)
        assert u.as_dict()["tokens_in"] == 12 and u.as_dict()["tokens_out"] == 5


class TestRetries:
    def test_429_is_retried_and_logged_as_a_recovery(self, ctx):
        recoveries = []
        a = Agent(FakeClient([Rate429(), Rate429(), GOOD_PAYLOAD]),
                  on_recovery=recoveries.append)
        assert a.draft(ctx).hypothesis
        assert len(recoveries) == 2
        assert all(r["recovery"] == "api_retry" for r in recoveries)
        assert all(r["event"] == "recovery" for r in recoveries)   # never an intervention

    def test_client_errors_are_not_retried(self, ctx):
        client = FakeClient([Bad400(), GOOD_PAYLOAD])
        with pytest.raises(AgentError):
            Agent(client).draft(ctx)
        assert len(client.calls) == 1, "a 400 is our bug; retrying just burns tokens"

    def test_gives_up_after_max_retries(self, ctx):
        client = FakeClient([Rate429()] * (agent_mod.MAX_RETRIES + 2))
        with pytest.raises(AgentError):
            Agent(client).draft(ctx)
        assert len(client.calls) == agent_mod.MAX_RETRIES

    def test_backoff_grows_and_is_jittered(self):
        delays = [agent_mod._backoff(i) for i in range(5)]
        assert delays[0] < delays[-1] <= 60
        assert len({round(agent_mod._backoff(3), 6) for _ in range(20)}) > 1


class TestRepair:
    def failed_node(self, tmp_path, **kw):
        ws = tmp_path / "n002"
        ws.mkdir()
        (ws / "pipeline.py").write_text("BROKEN_CODE_MARKER = (\n", encoding="utf-8")
        result = ExecResult(
            ok=False, exit_code=1, stdout_tail="training\n", stderr_tail="...",
            error_class="syntax", error_excerpt="SyntaxError: '(' was never closed",
            result_json=None, artifacts={}, wall_s=1.0, peak_rss_mb=10.0)
        return node(id="n002", kind="debug", workspace=ws, exec_result=result, **kw)

    def test_repair_prompt_carries_class_excerpt_and_code(self, ctx, tmp_path):
        a = Agent(FakeClient([]))
        a.repair(ctx, self.failed_node(tmp_path))
        user = a.client.calls[0]["messages"][0]["content"]
        assert "syntax" in user
        assert "'(' was never closed" in user
        assert "BROKEN_CODE_MARKER" in user

    def test_repair_reads_code_from_the_workspace_when_ctx_has_none(self, ctx, tmp_path):
        assert ctx.parent_code is None
        a = Agent(FakeClient([]))
        a.repair(ctx, self.failed_node(tmp_path))
        assert "BROKEN_CODE_MARKER" in a.client.calls[0]["messages"][0]["content"]

    def test_later_attempts_are_told_not_to_repeat_themselves(self, ctx, tmp_path):
        a = Agent(FakeClient([]))
        a.repair(ctx, self.failed_node(tmp_path, repair_attempts=2))
        user = a.client.calls[0]["messages"][0]["content"]
        assert "2 previous repair attempt(s) failed" in user
        assert "Do not repeat" in user

    def test_repair_without_an_exec_result_is_a_caller_bug(self, ctx):
        with pytest.raises(AgentError, match="exec_result"):
            Agent(FakeClient([])).repair(ctx, node())

    def test_three_attempts_then_dead(self, tmp_path):
        n = self.failed_node(tmp_path)
        assert not agent_mod.repair_exhausted(n)
        n.repair_attempts = agent_mod.MAX_REPAIR_ATTEMPTS
        assert agent_mod.repair_exhausted(n)


class TestSafety:
    def test_a_key_shaped_string_is_never_sent(self, ctx):
        ctx.data_card = "key sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345"
        with pytest.raises(AgentError, match="key-shaped"):
            Agent(FakeClient([])).draft(ctx)

    def test_improve_without_parent_code_is_a_caller_bug(self, ctx):
        with pytest.raises(AgentError, match="parent_code"):
            Agent(FakeClient([])).improve(ctx, node())

    def test_make_client_refuses_to_run_keyless(self, monkeypatch):
        # Every provider key has to go, not just Anthropic's: on a machine that happens to
        # export OPENAI_API_KEY this passed for the wrong reason and then failed once the
        # OpenAI adapter landed. The test is about having *no* key at all.
        no_keys(monkeypatch)
        with pytest.raises(AgentError, match="ANTHROPIC_API_KEY"):
            agent_mod.make_client()

    def test_make_client_honours_the_stub_switch(self, monkeypatch):
        monkeypatch.setenv("TECHJAM_LLM", "stub")
        assert isinstance(agent_mod.make_client(), StubClient)


class TestStubClient:
    def test_produces_a_contract_shaped_proposal(self, ctx):
        p = Agent(StubClient()).draft(ctx)
        assert p.hypothesis and p.code and p.tokens_in > 0
        assert "RESULT_JSON" in p.code
        assert "row_id,user_id,video_id,score" in p.code

    def test_stub_pipeline_actually_runs_and_satisfies_the_sandbox(self, ctx, tmp_path):
        from orchestrator import sandbox
        ws = tmp_path / "n000"
        ws.mkdir()
        data = tmp_path / "data"
        data.mkdir()
        (data / "val.csv").write_text("user_id,video_id\nu1,v1\nu1,v2\nu2,v1\n", encoding="utf-8")
        (ws / "pipeline.py").write_text(Agent(StubClient()).draft(ctx).code, encoding="utf-8")
        r = sandbox.run(node(workspace=ws), split="val", seed=0, timeout_s=30,
                        data_dir=data, mem_limit_mb=1024)
        assert r.ok, (r.error_class, r.error_excerpt)
        assert r.result_json["n_rows"] == 3


# --------------------------------------------------------------------------- #
# provider adapter
# --------------------------------------------------------------------------- #

class FakeOpenAIUsage:
    def __init__(self, prompt, completion, cached=0):
        self.prompt_tokens, self.completion_tokens = prompt, completion
        self.prompt_tokens_details = type("D", (), {"cached_tokens": cached})()


class FakeOpenAICompletion:
    def __init__(self, arguments, usage, name="submit_pipeline"):
        call = type("C", (), {"function": type("F", (), {
            "name": name, "arguments": arguments})()})()
        message = type("M", (), {"tool_calls": [call] if arguments is not None else None})()
        self.choices = [type("Ch", (), {"message": message})()]
        self.usage = usage


class FakeResponsesUsage:
    def __init__(self, input_tokens, output_tokens, cached=0):
        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        self.input_tokens_details = type("D", (), {"cached_tokens": cached})()


class FakeResponse:
    """The /v1/responses shape: a flat `output` list, never `choices`.

    A reasoning item is emitted before the call, as the real endpoint does — that is
    the whole reason for moving here, and the parser has to step over it.
    """

    def __init__(self, arguments, usage, name="submit_pipeline"):
        self.output = []
        if arguments is not None:
            self.output.append(type("R", (), {"type": "reasoning", "summary": []})())
            self.output.append(type("FC", (), {
                "type": "function_call", "name": name, "arguments": arguments})())
        self.usage = usage


class _FakeResponsesEndpoint:
    """`sdk.responses.create(...)`, recorded separately from chat.completions."""

    def __init__(self, sdk):
        self._sdk = sdk

    def create(self, **kwargs):
        self._sdk.responses_calls.append(kwargs)
        item = self._sdk.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeOpenAISDK:
    """Stands in for openai.OpenAI(). Records what the adapter sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.responses_calls = []

    @property
    def responses(self):
        return _FakeResponsesEndpoint(self)

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def openai_agent(script):
    from orchestrator.agent import OpenAIClient
    sdk = FakeOpenAISDK(script)
    return Agent(OpenAIClient(sdk), model="test-model"), sdk


@_needs_openai
class TestOpenAIAdapter:
    def ok_completion(self, **usage_kw):
        return FakeOpenAICompletion(
            json.dumps(GOOD_PAYLOAD),
            FakeOpenAIUsage(**{"prompt": 1000, "completion": 200, **usage_kw}))

    def test_proposal_comes_through_the_adapter_unchanged(self, ctx):
        a, _ = openai_agent([self.ok_completion()])
        p = a.draft(ctx)
        assert p.hypothesis.startswith("Active users dominate")
        assert p.plan == ["add the prior", "retrain", "write submission"]
        assert p.model == "test-model"

    def test_cached_tokens_are_not_double_counted(self, ctx):
        """OpenAI counts cached tokens inside prompt_tokens; Anthropic reports them
        separately. Summing both the same way would overstate a scored deliverable."""
        a, _ = openai_agent([self.ok_completion(prompt=1000, cached=800)])
        p = a.draft(ctx)
        assert p.tokens_in == 1000, "tokens_in must equal the API's own prompt_tokens"
        assert a.total.cache_read_input_tokens == 800
        assert a.total.input_tokens == 200

    def test_uncached_call_accounts_the_same(self, ctx):
        a, _ = openai_agent([self.ok_completion(prompt=1000, cached=0)])
        assert a.draft(ctx).tokens_in == 1000

    def test_system_block_is_flattened_and_cache_control_dropped(self, ctx):
        a, sdk = openai_agent([self.ok_completion()])
        a.draft(ctx)
        messages = sdk.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], str)
        assert "kuairand-pure" in messages[0]["content"]
        assert "cache_control" not in json.dumps(sdk.calls[0])
        assert messages[1]["role"] == "user"

    def test_static_prefix_still_comes_first(self, ctx):
        """OpenAI caches the longest matching prefix automatically, so the static
        block leading the request is what makes caching work at all."""
        a, sdk = openai_agent([self.ok_completion(), self.ok_completion()])
        a.draft(ctx)
        ctx.parent_code = "print(1)\n"
        a.improve(ctx, node())
        first, second = (c["messages"][0]["content"] for c in sdk.calls)
        assert first == second

    def test_tool_schema_is_translated(self, ctx):
        a, sdk = openai_agent([self.ok_completion()])
        a.draft(ctx)
        tool = sdk.calls[0]["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "submit_pipeline"
        assert tool["function"]["parameters"]["required"] == ["hypothesis", "plan", "code"]
        assert sdk.calls[0]["tool_choice"] == {
            "type": "function", "function": {"name": "submit_pipeline"}}

    def test_uses_max_completion_tokens(self, ctx):
        a, sdk = openai_agent([self.ok_completion()])
        a.draft(ctx)
        assert sdk.calls[0]["max_completion_tokens"] == agent_mod.DEFAULT_MAX_TOKENS
        assert "max_tokens" not in sdk.calls[0]

    def test_prose_reply_is_retried_like_any_other(self, ctx):
        prose = FakeOpenAICompletion(None, FakeOpenAIUsage(50, 10))
        recoveries = []
        from orchestrator.agent import OpenAIClient
        sdk = FakeOpenAISDK([prose, self.ok_completion()])
        a = Agent(OpenAIClient(sdk), model="test-model", on_recovery=recoveries.append)
        assert a.draft(ctx).hypothesis
        assert recoveries[0]["recovery"] == "no_tool_use"

    def test_unparseable_tool_arguments_are_retried_not_crashed(self, ctx):
        broken = FakeOpenAICompletion("{not json", FakeOpenAIUsage(50, 10))
        a, _ = openai_agent([broken, self.ok_completion()])
        assert a.draft(ctx).hypothesis

    def test_rate_limits_retry_through_the_adapter(self, ctx):
        a, sdk = openai_agent([Rate429(), self.ok_completion()])
        assert a.draft(ctx).hypothesis
        assert len(sdk.calls) == 2


@_needs_openai
class TestProviderSelection:
    @_needs_anthropic
    def test_anthropic_key_wins_when_both_are_present(self, monkeypatch):
        monkeypatch.delenv("TECHJAM_LLM", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-x")
        client = agent_mod.make_client()
        assert type(client).__name__ == "Anthropic"

    def test_falls_back_to_openai_when_that_is_the_only_key(self, monkeypatch):
        monkeypatch.delenv("TECHJAM_LLM", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-x")
        assert isinstance(agent_mod.make_client(), agent_mod.OpenAIClient)

    def test_explicit_provider_overrides_the_environment(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-x")
        assert isinstance(agent_mod.make_client(provider="openai"), agent_mod.OpenAIClient)

    def test_no_key_at_all_names_both_options(self, monkeypatch):
        no_keys(monkeypatch)
        with pytest.raises(AgentError, match="ANTHROPIC_API_KEY or OPENAI_API_KEY"):
            agent_mod.make_client()

    def test_unknown_provider_is_rejected(self, monkeypatch):
        monkeypatch.setenv("TECHJAM_LLM", "llama")
        with pytest.raises(AgentError, match="unknown provider"):
            agent_mod.make_client()

    def test_openai_model_must_be_named_not_guessed(self, monkeypatch):
        """A wrong model id discovered four hours into a scored run is a lost run."""
        monkeypatch.delenv("TECHJAM_MODEL", raising=False)
        client = agent_mod.OpenAIClient(FakeOpenAISDK([]))
        with pytest.raises(AgentError, match="TECHJAM_MODEL"):
            agent_mod.default_model_for(client)

    def test_openai_model_from_env(self, monkeypatch):
        monkeypatch.setenv("TECHJAM_MODEL", "some-model")
        client = agent_mod.OpenAIClient(FakeOpenAISDK([]))
        assert agent_mod.default_model_for(client) == "some-model"

    def test_anthropic_default_needs_no_configuration(self, monkeypatch):
        monkeypatch.delenv("TECHJAM_MODEL", raising=False)
        assert agent_mod.default_model_for(FakeClient([])) == agent_mod.DEFAULT_MODEL


class TestDotenv:
    def test_reads_keys_without_overriding_the_environment(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text(
            "# a comment\n\n"
            "OPENAI_API_KEY=sk-proj-from-file\n"
            'export TECHJAM_MODEL="some-model"\n'
            "ALREADY_SET=from-file\n", encoding="utf-8")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("TECHJAM_MODEL", raising=False)
        monkeypatch.setenv("ALREADY_SET", "from-shell")
        agent_mod.load_dotenv(env)
        assert os.environ["OPENAI_API_KEY"] == "sk-proj-from-file"
        assert os.environ["TECHJAM_MODEL"] == "some-model"
        assert os.environ["ALREADY_SET"] == "from-shell", "the shell wins over .env"

    def test_missing_file_is_not_an_error(self, tmp_path):
        agent_mod.load_dotenv(tmp_path / "nope.env")   # must not raise

    def test_malformed_lines_are_skipped(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("no equals sign here\nGOOD_ONE=yes\n", encoding="utf-8")
        monkeypatch.delenv("GOOD_ONE", raising=False)
        agent_mod.load_dotenv(env)
        assert os.environ["GOOD_ONE"] == "yes"


@_needs_openai
class TestKeyShape:
    def test_openai_key_under_the_anthropic_variable_is_caught(self, monkeypatch):
        """Exactly the mistake that costs an hour: right key, wrong variable."""
        monkeypatch.setenv("TECHJAM_LLM", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-proj-abcdefghijklmnop")
        with pytest.raises(AgentError, match="looks like an OpenAI key"):
            agent_mod.make_client()

    def test_anthropic_key_under_the_openai_variable_is_caught(self, monkeypatch):
        monkeypatch.setenv("TECHJAM_LLM", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-ant-abcdefghijklmnop")
        with pytest.raises(AgentError, match="looks like an Anthropic key"):
            agent_mod.make_client()

    def test_a_correctly_placed_key_passes(self, monkeypatch):
        monkeypatch.setenv("TECHJAM_LLM", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijklmnop")
        assert isinstance(agent_mod.make_client(), agent_mod.OpenAIClient)


class TestTemperatureFallback:
    """Reasoning models accept only the default temperature. A 400 is not retryable,
    so without this the first improve() call of a run would kill its node."""

    class Temp400(Exception):
        status_code = 400

        def __str__(self):
            return ("Unsupported value: 'temperature' does not support 0.6 with this "
                    "model. Only the default (1) value is supported.")

    class Other400(Exception):
        status_code = 400

        def __str__(self):
            return "Invalid schema for function 'submit_pipeline'."

    def completion(self):
        return FakeOpenAICompletion(json.dumps(GOOD_PAYLOAD), FakeOpenAIUsage(900, 100))

    def test_retries_without_temperature_and_succeeds(self, ctx):
        a, sdk = openai_agent([self.Temp400(), self.completion()])
        assert a.draft(ctx).hypothesis
        assert "temperature" in sdk.calls[0]
        assert "temperature" not in sdk.calls[1]
        assert sdk.calls[1]["model"] == "test-model"

    def test_the_lesson_is_remembered_per_model(self, ctx):
        """Paying the 400 once per run is fine; paying it on all 50 iterations is not."""
        a, sdk = openai_agent([self.Temp400(), self.completion(), self.completion()])
        a.draft(ctx)
        ctx.parent_code = "print(1)\n"
        a.improve(ctx, node())
        assert len(sdk.calls) == 3
        assert "temperature" not in sdk.calls[2], "should not re-learn the same lesson"
        assert a.client.fixed_temperature_models() == {"test-model"}

    def test_other_400s_are_not_swallowed(self, ctx):
        a, sdk = openai_agent([self.Other400(), self.completion()])
        with pytest.raises(AgentError):
            a.draft(ctx)
        assert len(sdk.calls) == 1, "only a temperature 400 gets the second chance"

    def test_a_model_that_accepts_temperature_is_left_alone(self, ctx):
        a, sdk = openai_agent([self.completion()])
        a.draft(ctx)
        assert sdk.calls[0]["temperature"] == agent_mod.TEMPERATURE["draft"]
        assert a.client.fixed_temperature_models() == set()


class TestResponsesApiFallback:
    """gpt-5.6-class models refuse function tools + reasoning on chat.completions.

    The API offers two ways out and they are not equivalent: `reasoning_effort="none"`
    clears the 400 by switching off the reasoning this agent exists to use. Moving the
    model to /v1/responses keeps both. A live gpt-5.6-terra dev run made 0 LLM calls
    across 8 iterations before this existed, and still exited 0.
    """

    class Reasoning400(Exception):
        status_code = 400

        def __str__(self):
            return ("Function tools with reasoning_effort are not supported for "
                    "gpt-5.6-terra in /v1/chat/completions. To use function tools, "
                    "use /v1/responses or set reasoning_effort to 'none'.")

    def response(self, **usage_kw):
        return FakeResponse(
            json.dumps(GOOD_PAYLOAD),
            FakeResponsesUsage(**{"input_tokens": 900, "output_tokens": 100, **usage_kw}),
        )

    def test_it_moves_to_the_responses_api_and_succeeds(self, ctx):
        a, sdk = openai_agent([self.Reasoning400(), self.response()])
        assert a.draft(ctx).hypothesis.startswith("Active users dominate")
        assert len(sdk.calls) == 1, "one rejected chat.completions attempt"
        assert len(sdk.responses_calls) == 1, "then the same call on /v1/responses"

    def test_reasoning_is_kept_rather_than_switched_off(self, ctx):
        """The point of the endpoint move. Setting it to 'none' would also pass."""
        a, sdk = openai_agent([self.Reasoning400(), self.response()])
        a.draft(ctx)
        assert sdk.responses_calls[0].get("reasoning_effort") != "none"
        assert all(c.get("reasoning_effort") != "none" for c in sdk.calls)

    def test_the_lesson_is_remembered_per_model(self, ctx):
        """Paying the 400 once a run is fine; paying it on all 50 iterations is not."""
        a, sdk = openai_agent([self.Reasoning400(), self.response(), self.response()])
        a.draft(ctx)
        ctx.parent_code = "print(1)\n"
        a.improve(ctx, node())
        assert len(sdk.calls) == 1, "the second call must skip chat.completions entirely"
        assert len(sdk.responses_calls) == 2
        assert a.client.responses_api_models() == {"test-model"}

    def test_the_tool_is_sent_flat_not_nested(self, ctx):
        a, sdk = openai_agent([self.Reasoning400(), self.response()])
        a.draft(ctx)
        tool = sdk.responses_calls[0]["tools"][0]
        assert tool["name"] == "submit_pipeline", "responses puts name at the top level"
        assert "function" not in tool, "that nesting is the chat.completions shape"
        assert "strict" not in tool, (
            "strict requires additionalProperties:false and every property required; "
            "PROPOSAL_TOOL declares neither, so it would trade one 400 for another"
        )

    def test_the_system_prompt_becomes_instructions(self, ctx):
        a, sdk = openai_agent([self.Reasoning400(), self.response()])
        a.draft(ctx)
        call = sdk.responses_calls[0]
        assert "kuairand-pure" in call["instructions"]
        assert "cache_control" not in json.dumps(call, default=str)

    def test_cached_tokens_are_not_double_counted(self, ctx):
        """`input_tokens` includes cached reads here too, exactly as prompt_tokens does."""
        a, _ = openai_agent([self.Reasoning400(),
                             self.response(input_tokens=1000, cached=800)])
        p = a.draft(ctx)
        assert p.tokens_in == 1000
        assert a.total.cache_read_input_tokens == 800
        assert a.total.input_tokens == 200

    def test_reasoning_output_tokens_are_still_billed(self, ctx):
        """They are real spend and Feasibility is scored on tokens."""
        a, _ = openai_agent([self.Reasoning400(),
                             self.response(output_tokens=4321)])
        assert a.draft(ctx).tokens_out == 4321

    def test_other_400s_are_not_swallowed(self, ctx):
        a, sdk = openai_agent([TestTemperatureFallback.Other400(), self.response()])
        with pytest.raises(AgentError):
            a.draft(ctx)
        assert sdk.responses_calls == [], "only a reasoning 400 moves endpoint"


# --------------------------------------------------------------------------- #
# the offline loop
#
# The stub pipeline used to write fabricated ids (`row_id % 97`, `row_id % 313`) for a
# fixed row count, which can align with no real evaluation split. Every stubbed run
# therefore died on "submission failed validation: line 2 misaligned", three repairs and
# a dead node — so the loop had never produced a scored node on real data, and nothing
# downstream of the sandbox had ever been exercised end to end. These tests exist so that
# cannot come back quietly.
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
KUAIRAND_DIR = REPO_ROOT / "data" / "KuaiRand-Pure" / "data"
needs_kuairand = pytest.mark.skipif(
    not (KUAIRAND_DIR / "log_standard_4_08_to_4_21_pure.csv").is_file(),
    reason="needs the KuaiRand-Pure download; see the README for the curl line",
)

DEMO_TASK = TaskSpec(
    name="demo", data_dir=Path("data"), metrics=("rmse",),
    baseline_val={}, baseline_test={},
    submission_columns=("row_id", "listing_id", "prediction"),
    prediction_column="prediction",
)


def task_spec_for(cfg, **overrides):
    """The `TaskSpec` run.py builds from a task file, without run.py's argparse."""
    fields = {
        "name": cfg.name, "data_dir": cfg.data.dir, "metrics": cfg.report_metrics,
        "baseline_val": dict(cfg.baseline_val), "baseline_test": dict(cfg.baseline_test),
        "ceiling": cfg.ceiling, "kind": cfg.kind, "description": cfg.description,
        "primary_parts": cfg.primary_parts,
        "submission_columns": cfg.submission_columns,
        "prediction_column": cfg.prediction_column, "seed_std": cfg.seed_std,
        "config": cfg,
    }
    fields.update(overrides)
    return TaskSpec(**fields)


def tiny_regression_task(tmp_path, name="stub-align"):
    """A complete generic task on disk: config, data, and the TaskSpec the loop uses."""
    import csv

    from orchestrator import taskspec as ts

    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"listing_id": i, "rooms": i % 5, "area": 30.0 + i, "rent": 500.0 + 12.5 * i}
        for i in range(120)
    ]
    with open(tmp_path / "all.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["listing_id", "rooms", "area", "rent"])
        writer.writeheader()
        writer.writerows(rows)

    cfg = ts.parse_task({
        "name": name,
        "kind": "regression",
        "description": "Predict rent from rooms and area.",
        "data": {
            "dir": str(tmp_path),
            "file": "all.csv",
            "target": "rent",
            "id_columns": ["listing_id"],
            "split": {"strategy": "random", "valid_frac": 0.2, "test_frac": 0.2, "seed": 0},
        },
        "submission": {"columns": ["row_id", "listing_id", "prediction"]},
        "metrics": {"primary": ["rmse"], "report": ["rmse", "mae"]},
    })
    return cfg, task_spec_for(cfg, max_iters=2, wall_clock_s=300)


def run_pipeline(code, workspace, data_dir, split, columns):
    from orchestrator import sandbox

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "pipeline.py").write_text(code, encoding="utf-8")
    return sandbox.run(
        node(workspace=workspace), split=split, seed=0, timeout_s=180,
        data_dir=Path(data_dir), mem_limit_mb=4096, header_columns=tuple(columns),
    )


class TestStubPipelineAlignment:
    def test_it_aligns_with_a_generic_tabular_split(self, tmp_path):
        from orchestrator import datasource as ds
        from orchestrator import evaluate
        from tests.stubs.agent import render_pipeline, spec_for

        cfg, spec = tiny_regression_task(tmp_path / "data")
        code = render_pipeline("align", spec=spec_for(spec))
        imported = {
            line.split()[1].split(".")[0]
            for line in code.splitlines()
            if line.startswith(("import ", "from "))
        }
        assert "orchestrator" not in imported, "a pipeline may not import from us"

        for split in ("val", "test"):
            result = run_pipeline(code, tmp_path / f"ws-{split}", ds.materialise(cfg),
                                  split, cfg.submission_columns)
            assert result.ok, (result.error_class, result.error_excerpt)
            ok, message = evaluate.validate(result.artifacts["submission"], split, cfg)
            assert ok, message

    @needs_kuairand
    def test_it_aligns_with_the_real_kuairand_splits(self, tmp_path):
        """Row order here is `data.load()` order, not something that merely looks like
        it: the validator compares (user_id, video_id) line by line, and 3.06% of test
        rows are duplicate pairs, so only the true order survives."""
        from orchestrator import evaluate
        from orchestrator.taskspec import load_task
        from tests.stubs.agent import render_pipeline, spec_for

        cfg = load_task("kuairand-pure")
        code = render_pipeline("align", spec=spec_for(task_spec_for(cfg)))

        for split, expected in (("val", 124_909), ("test", 170_588)):
            result = run_pipeline(code, tmp_path / f"ws-{split}", KUAIRAND_DIR, split,
                                  cfg.submission_columns)
            assert result.ok, (result.error_class, result.error_excerpt)
            assert result.result_json["n_rows"] == expected
            ok, message = evaluate.validate(result.artifacts["submission"], split, cfg)
            assert ok, message


def offline_run(tmp_path, agent, *, max_iters=2, run_id="roffline"):
    """One real loop: canned proposals, but the real sandbox and the real evaluator."""
    from dataclasses import replace

    from orchestrator import evaluate, sandbox
    from orchestrator import journal as journal_mod
    from orchestrator.core import Orchestrator

    cfg, spec = tiny_regression_task(tmp_path / "data", name=f"offline-{run_id}")
    run_dir = tmp_path / run_id
    journal_mod.close()
    orch = Orchestrator(
        replace(spec, max_iters=max_iters),
        run_dir=run_dir, run_id=run_id, agent=agent,
        executor=sandbox, evaluator=evaluate, mode="smoke", timeout_s=180,
        journal=journal_mod.Journal(run_dir / "journal.jsonl", run_id, fsync=False),
    )
    summary = orch.run()
    journal_mod.close()
    return cfg, summary


class TestOfflineLoopProducesAScoredNode:
    """The acceptance test for this whole change: `--agent stub --sandbox auto
    --evaluator auto` must end with a best node and a valid final submission."""

    def test_two_iterations_score_and_finalise(self, tmp_path):
        from orchestrator import evaluate
        from tests.stubs import StubAgent

        cfg, summary = offline_run(tmp_path, StubAgent())
        assert summary["best_node"] is not None, summary
        assert summary["best_metrics"] is not None
        assert summary["final_valid"] is True
        final = Path(summary["final_submission"])
        assert final.is_file() and final.name == "submission.csv"
        ok, message = evaluate.validate(final, "test", cfg)
        assert ok, message

    def test_the_replay_agent_drives_the_same_loop_with_no_key(self, tmp_path, monkeypatch):
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TECHJAM_LLM"):
            monkeypatch.delenv(var, raising=False)
        _cfg, summary = offline_run(tmp_path, agent_mod.ReplayAgent(),
                                    max_iters=3, run_id="rreplay")
        assert summary["best_node"] is not None, summary
        assert summary["final_valid"] is True
        assert summary["tokens_total"] == 0, "replay must not report spend it did not incur"


class TestReplayAgent:
    def serve(self, tmp_path, bodies):
        for i, body in enumerate(bodies):
            (tmp_path / f"{i:02d}_p.py").write_text(body, encoding="utf-8")
        return agent_mod.ReplayAgent(directory=tmp_path)

    def test_the_shipped_pipelines_parse_and_stay_task_agnostic(self):
        for path in agent_mod.ReplayAgent().paths:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            assert agent_mod.REPLAY_HEADER_TOKEN in source, f"{path.name} hardcodes a header"

    def test_serves_each_file_once_then_wraps(self, ctx, tmp_path):
        replay = self.serve(tmp_path, ['"""A."""\n', '"""B."""\n'])
        served = [replay.draft(ctx).code, replay.improve(ctx, node()).code,
                  replay.repair(ctx, node()).code]
        assert served[0].startswith('"""A.')
        assert served[1].startswith('"""B.')
        assert served[2].startswith('"""A.'), "must wrap round rather than run out"
        assert replay.calls == ["draft", "improve", "repair"]

    def test_substitutes_the_tasks_submission_header(self, ctx, tmp_path):
        replay = self.serve(
            tmp_path, [f'"""Doc."""\nHEADER = "{agent_mod.REPLAY_HEADER_TOKEN}"\n'])
        ctx.task = DEMO_TASK
        code = replay.draft(ctx).code
        assert 'HEADER = "row_id,listing_id,prediction"' in code
        assert agent_mod.REPLAY_HEADER_TOKEN not in code

    def test_reports_no_spend_and_carries_the_files_own_reasoning(self, ctx, tmp_path):
        replay = self.serve(
            tmp_path, ['"""Rank by popularity.\n\nBecause exposure is skewed."""\n'])
        proposal = replay.draft(ctx)
        assert proposal.tokens_in == 0 and proposal.tokens_out == 0
        assert "Rank by popularity." in proposal.hypothesis
        assert "Because exposure is skewed." in proposal.hypothesis
        assert proposal.plan and proposal.model.startswith("replay-agent-v1:")

    def test_an_empty_directory_is_a_loud_error(self, tmp_path):
        with pytest.raises(AgentError, match="no canned pipelines"):
            agent_mod.ReplayAgent(directory=tmp_path)

    def test_get_agent_honours_the_replay_switch(self, monkeypatch):
        monkeypatch.setattr(agent_mod, "_DEFAULT", None)
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv(agent_mod.REPLAY_ENV, "replay")
        assert isinstance(agent_mod.get_agent(), agent_mod.ReplayAgent)
        monkeypatch.setattr(agent_mod, "_DEFAULT", None)


# --------------------------------------------------------------------------- #
# Gemini adapter and the provider chain
#
# Nothing here touches the network. `GeminiClient` takes its transport as an argument
# precisely so the tests can hand it a recorder: a suite that could reach an API is a
# suite that can bill us and flake on a train.
# --------------------------------------------------------------------------- #

GEMINI_OK = {
    "candidates": [{
        "content": {
            "role": "model",
            "parts": [{"functionCall": {"name": "submit_pipeline", "args": GOOD_PAYLOAD}}],
        },
        "finishReason": "STOP",
    }],
    "usageMetadata": {
        "promptTokenCount": 1000, "candidatesTokenCount": 300,
        "cachedContentTokenCount": 400, "thoughtsTokenCount": 50,
    },
}
GEMINI_PROSE = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "Here is my plan."}]}}],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
}


class FakeGeminiTransport:
    """Stands in for one HTTPS round trip. `script` is payloads or exceptions."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": body,
                           "timeout": timeout})
        item = self.script.pop(0) if self.script else GEMINI_OK
        if isinstance(item, Exception):
            raise item
        return item


def gemini_client(script=None, key="AQ.Ab8RN6J-fake-key-shape-that-is-not-AIza"):
    transport = FakeGeminiTransport(script)
    return agent_mod.GeminiClient(key, transport=transport), transport


def gemini_agent(script=None, **kwargs):
    client, transport = gemini_client(script)
    return Agent(client, model="gemini-2.5-pro", **kwargs), transport


class TestGeminiAdapter:
    def test_the_key_travels_in_a_header_and_never_in_the_url(self, ctx):
        agent, transport = gemini_agent()
        agent.draft(ctx)
        call = transport.calls[0]
        assert call["headers"]["x-goog-api-key"].startswith("AQ.")
        assert "key" not in call["url"] and "AQ." not in call["url"], \
            "a key in a query string is a key in every proxy log between here and Google"
        assert call["url"].endswith("/models/gemini-2.5-pro:generateContent")

    def test_the_system_block_becomes_a_system_instruction(self, ctx):
        agent, transport = gemini_agent()
        agent.draft(ctx)
        body = transport.calls[0]["body"]
        text = body["systemInstruction"]["parts"][0]["text"]
        assert "kuairand-pure" in text and "long_view base rate 0.42" in text
        assert body["contents"][0]["role"] == "user"
        assert "kuairand-pure" not in body["contents"][0]["parts"][0]["text"], \
            "the static block must not also ride in the turn, or it is billed twice"

    def test_the_proposal_tool_is_declared_and_forced(self, ctx):
        agent, transport = gemini_agent()
        agent.draft(ctx)
        body = transport.calls[0]["body"]
        declared = body["tools"][0]["functionDeclarations"][0]
        assert declared["name"] == "submit_pipeline"
        assert declared["parameters"]["type"] == "OBJECT", "Schema.type is an enum name"
        assert declared["parameters"]["properties"]["plan"]["type"] == "ARRAY"
        assert declared["parameters"]["properties"]["plan"]["items"]["type"] == "STRING"
        assert set(declared["parameters"]["required"]) == {"hypothesis", "plan", "code"}
        config = body["toolConfig"]["functionCallingConfig"]
        assert config["mode"] == "ANY"
        assert config["allowedFunctionNames"] == ["submit_pipeline"]

    def test_generation_config_carries_the_budget_and_temperature(self, ctx):
        agent, transport = gemini_agent()
        agent.draft(ctx)
        config = transport.calls[0]["body"]["generationConfig"]
        assert config["maxOutputTokens"] == agent_mod.DEFAULT_MAX_TOKENS
        assert config["temperature"] == agent_mod.TEMPERATURE["draft"]

    def test_a_function_call_comes_back_as_a_proposal(self, ctx):
        agent, _ = gemini_agent()
        proposal = agent.draft(ctx)
        assert proposal.hypothesis == GOOD_PAYLOAD["hypothesis"]
        assert proposal.plan == GOOD_PAYLOAD["plan"]
        assert proposal.code.strip() == "print('hello')"
        assert proposal.model == "gemini-2.5-pro"

    def test_usage_accounting_is_exact(self, ctx):
        """promptTokenCount includes the cached prefix, exactly like OpenAI's
        prompt_tokens. Summing it with the cache read would bill us twice for it in a
        number we report under Feasibility."""
        agent, _ = gemini_agent()
        agent.draft(ctx)
        usage = agent.total
        assert usage.input_tokens == 600          # 1000 prompt - 400 cached
        assert usage.cache_read_input_tokens == 400
        assert usage.tokens_in == 1000            # and the total is still the truth
        assert usage.output_tokens == 350         # 300 answer + 50 reasoning
        assert usage.calls == 1

    def test_missing_usage_metadata_is_zero_not_a_crash(self):
        assert agent_mod._as_gemini_usage(None).calls == 1
        assert agent_mod._as_gemini_usage({}).tokens_in == 0

    def test_a_prose_reply_is_retried_not_accepted(self, ctx):
        agent, transport = gemini_agent([GEMINI_PROSE, GEMINI_OK])
        assert agent.draft(ctx).hypothesis == GOOD_PAYLOAD["hypothesis"]
        assert len(transport.calls) == 2

    def test_a_refusal_is_reported_rather_than_retried_five_times(self, ctx):
        blocked = {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
        agent, transport = gemini_agent([blocked])
        with pytest.raises(AgentError, match="finishReason=SAFETY"):
            agent.draft(ctx)
        assert len(transport.calls) == 1

    def test_a_truncated_reply_is_retried_because_a_retry_can_fit(self, ctx):
        """Reasoning length varies between attempts, so MAX_TOKENS is not a refusal."""
        truncated = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
        agent, transport = gemini_agent([truncated, GEMINI_OK])
        assert agent.draft(ctx).hypothesis == GOOD_PAYLOAD["hypothesis"]
        assert len(transport.calls) == 2

    def test_a_blocked_prompt_says_so(self, ctx):
        agent, _ = gemini_agent([{"promptFeedback": {"blockReason": "OTHER"}}])
        with pytest.raises(AgentError, match="blockReason=OTHER"):
            agent.draft(ctx)

    def test_an_http_error_carries_its_status_for_classification(self):
        error = agent_mod.GeminiAPIError(429, "Resource exhausted")
        assert error.status_code == 429
        assert agent_mod._retryable(error) and not agent_mod._provider_fatal(error)
        assert agent_mod._provider_fatal(agent_mod.GeminiAPIError(404, "not found"))

    def test_list_models_strips_the_models_prefix(self):
        client, transport = gemini_client()
        transport.script = [{"models": [{"name": "models/gemini-2.5-pro"},
                                        {"name": "models/gemini-2.5-flash"}]}]
        assert client.list_models() == ["gemini-2.5-flash", "gemini-2.5-pro"]
        assert transport.calls[0]["body"] is None, "listing is a GET, not a POST"


class TestGeminiKeyAndModelConfig:
    def test_an_unfamiliar_key_shape_is_accepted(self, monkeypatch):
        """Google issues more than one key shape; ours starts `AQ.`, not `AIza`. A rule
        that guessed the format would reject a working key, which is worse than the
        mistake it was meant to catch."""
        no_keys(monkeypatch)
        monkeypatch.setenv("TECHJAM_LLM", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "AQ.Ab8RN6J" + "x" * 44)
        assert isinstance(agent_mod.make_client(), agent_mod.GeminiClient)

    def test_google_api_key_is_accepted_as_an_alias(self, monkeypatch):
        no_keys(monkeypatch)
        monkeypatch.setenv("TECHJAM_LLM", "gemini")
        monkeypatch.setenv("GOOGLE_API_KEY", "AQ.whatever")
        assert isinstance(agent_mod.make_client(), agent_mod.GeminiClient)

    def test_another_vendors_key_under_the_gemini_variable_is_caught(self, monkeypatch):
        no_keys(monkeypatch)
        monkeypatch.setenv("TECHJAM_LLM", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "sk-proj-abcdefghijklmnop")
        with pytest.raises(AgentError, match="Anthropic or OpenAI key"):
            agent_mod.make_client()

    def test_gemini_is_auto_detected_when_it_is_the_only_key(self, monkeypatch):
        no_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "AQ.only-key")
        assert isinstance(agent_mod.make_client(), agent_mod.GeminiClient)

    def test_the_model_must_be_named_not_guessed(self, monkeypatch):
        monkeypatch.delenv("TECHJAM_MODEL", raising=False)
        client, _ = gemini_client()
        with pytest.raises(AgentError, match="TECHJAM_MODEL"):
            agent_mod.default_model_for(client)

    def test_the_fallback_reads_its_own_model_variable(self, monkeypatch):
        monkeypatch.delenv("TECHJAM_MODEL", raising=False)
        monkeypatch.setenv("TECHJAM_FALLBACK_MODEL", "gemini-2.5-flash")
        client, _ = gemini_client()
        assert agent_mod.default_model_for(
            client, env="TECHJAM_FALLBACK_MODEL") == "gemini-2.5-flash"


class Throttled(Exception):
    status_code = 429


class Unauthorized(Exception):
    status_code = 401


class NoSuchModel(Exception):
    status_code = 404


def openai_completion(payload=None):
    return FakeOpenAICompletion(json.dumps(payload or GOOD_PAYLOAD),
                                FakeOpenAIUsage(1000, 200))


def chained(primary_script, *, fallback_script=None, on_recovery=None):
    """An Agent with a scripted OpenAI primary and a scripted Gemini fallback.

    The primary is a real `OpenAIClient` over a fake SDK, not a bare fake, so the
    provider names in the journal are the ones a real run would carry.
    """
    sdk = FakeOpenAISDK(primary_script)
    fallback, transport = gemini_client(fallback_script)
    agent = Agent(agent_mod.OpenAIClient(sdk), model="gpt-4o", fallback=fallback,
                  fallback_model="gemini-2.5-pro", on_recovery=on_recovery)
    return agent, sdk, transport


class TestProviderFallback:
    def test_no_fallback_is_configured_by_default(self, ctx):
        agent = Agent(FakeClient([]), model="gpt-4o")
        assert agent.fallback is None
        assert len(agent.providers) == 1

    def test_a_throttled_primary_falls_over_and_the_run_continues(self, ctx):
        agent, sdk, transport = chained([Throttled()] * agent_mod.MAX_RETRIES)
        proposal = agent.draft(ctx)
        assert proposal.hypothesis == GOOD_PAYLOAD["hypothesis"]
        assert proposal.model == "gemini-2.5-pro", "the journal must name who answered"
        assert len(sdk.calls) == agent_mod.MAX_RETRIES, "backoff first, then fail over"
        assert len(transport.calls) == 1

    def test_a_throttled_primary_is_not_written_off_for_the_whole_run(self, ctx):
        """Throttling passes. Disabling the primary after one busy minute would spend
        the rest of a six-hour run on the fallback for no reason."""
        agent, sdk, transport = chained(
            [Throttled()] * agent_mod.MAX_RETRIES + [openai_completion()])
        agent.draft(ctx)
        assert agent.providers[0].disabled is False
        ctx.parent_code = "print(1)\n"
        assert agent.improve(ctx, node()).model == "gpt-4o", "primary tried again"
        assert len(sdk.calls) == agent_mod.MAX_RETRIES + 1
        assert len(transport.calls) == 1

    @pytest.mark.parametrize("failure", [Unauthorized(), NoSuchModel()])
    def test_auth_and_missing_model_disable_the_primary_permanently(self, ctx, failure):
        agent, sdk, transport = chained([failure])
        agent.draft(ctx)
        assert agent.providers[0].disabled is True, "a bad key will not fix itself"
        assert len(sdk.calls) == 1, "no backoff on a 401/404 — it is not transient"
        ctx.parent_code = "print(1)\n"
        agent.improve(ctx, node())
        assert len(sdk.calls) == 1, "the dead primary must not be tried again"
        assert len(transport.calls) == 2
        report = agent.provider_report()
        assert report["primary_dead"] is True and report["fallback_calls"] == 2

    def test_a_malformed_proposal_never_triggers_a_failover(self, ctx):
        """This is the line that matters. A bad answer is the repair loop's problem;
        swapping providers over it would hide a real quality problem behind a switch."""
        prose = FakeOpenAICompletion(None, FakeOpenAIUsage(100, 10))
        agent, sdk, transport = chained([prose] * agent_mod.MAX_RETRIES)
        with pytest.raises(AgentError, match="never returned a submit_pipeline"):
            agent.draft(ctx)
        assert transport.calls == [], "the fallback must not have been reached"
        assert len(sdk.calls) == agent_mod.MAX_RETRIES

    def test_an_empty_hypothesis_never_triggers_a_failover(self, ctx):
        empty = openai_completion({**GOOD_PAYLOAD, "hypothesis": "  "})
        agent, _sdk, transport = chained([empty, empty])
        with pytest.raises(AgentError, match="empty hypothesis"):
            agent.draft(ctx)
        assert transport.calls == []

    def test_a_dead_fallback_raises_rather_than_looping(self, ctx):
        agent, _sdk, _t = chained(
            [Unauthorized()],
            fallback_script=[agent_mod.GeminiAPIError(401, "API key not valid")])
        with pytest.raises(agent_mod.ProviderError):
            agent.draft(ctx)

    def test_the_switch_is_announced_with_both_providers_and_the_reason(self, ctx):
        seen = []
        agent, _sdk, transport = chained([Throttled()] * agent_mod.MAX_RETRIES,
                                         on_recovery=seen.append)
        agent.draft(ctx)
        switch = [e for e in seen if e["recovery"] == "provider_failover"]
        assert len(switch) == 1
        event = switch[0]
        assert event["from_provider"] == "openai:gpt-4o"
        assert event["to_provider"] == "gemini:gemini-2.5-pro"
        assert "Throttled" in event["error_excerpt"]
        assert event["primary_disabled"] is False, "a 429 is not a dead provider"
        # The event fires before the retry, so it carries the split as it stood at the
        # switch; the fallback's own call lands right after it.
        assert event["providers"]["openai:gpt-4o"]["calls"] == 0
        assert len(transport.calls) == 1
        assert agent.provider_report()["fallback_calls"] == 1

    def test_the_report_splits_calls_and_tokens_by_provider(self, ctx):
        agent, _sdk, _t = chained([openai_completion()])
        agent.draft(ctx)
        report = agent.provider_report()
        assert report["primary"] == "openai:gpt-4o"
        assert report["fallback"] == "gemini:gemini-2.5-pro"
        assert report["providers"]["openai:gpt-4o"]["calls"] == 1
        assert report["providers"]["openai:gpt-4o"]["tokens_in"] == 1000
        assert report["providers"]["gemini:gemini-2.5-pro"]["calls"] == 0
        assert report["fallback_calls"] == 0 and report["primary_dead"] is False

    def test_an_ambient_fallback_is_not_smuggled_into_an_explicit_client(self, monkeypatch):
        """A caller that hands us a client is running that provider on purpose."""
        monkeypatch.setenv("TECHJAM_FALLBACK_LLM", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "AQ.whatever")
        monkeypatch.setenv("TECHJAM_FALLBACK_MODEL", "gemini-2.5-pro")
        assert Agent(FakeClient([]), model="m").fallback is None

    def test_an_unusable_fallback_does_not_stop_the_run(self, monkeypatch):
        no_keys(monkeypatch)
        monkeypatch.setenv("TECHJAM_FALLBACK_LLM", "gemini")   # configured, but no key
        assert agent_mod._fallback_provider() is None

    def test_a_configured_fallback_is_built_from_the_environment(self, monkeypatch):
        no_keys(monkeypatch)
        monkeypatch.setenv("TECHJAM_FALLBACK_LLM", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "AQ.whatever")
        monkeypatch.setenv("TECHJAM_FALLBACK_MODEL", "gemini-2.5-pro")
        provider = agent_mod._fallback_provider()
        assert provider is not None
        assert provider.name == "gemini" and provider.model == "gemini-2.5-pro"


class TestStubClientPipeline:
    def test_it_reads_the_split_named_in_the_prompts_header(self, ctx, tmp_path):
        """`StubClient` is handed a prompt, never a TaskSpec, so the submission header
        has to be recovered from the prompt. Hardcoding KuaiRand's broke every other
        task before the sandbox's structural check had even finished."""
        from orchestrator import sandbox

        ctx.task = DEMO_TASK
        code = Agent(StubClient()).draft(ctx).code
        assert "row_id,listing_id,prediction" in code

        data = tmp_path / "data"
        data.mkdir()
        (data / "valid.csv").write_text("listing_id,area\n7,30\n8,40\n9,50\n",
                                        encoding="utf-8")
        ws = tmp_path / "n000"
        ws.mkdir()
        (ws / "pipeline.py").write_text(code, encoding="utf-8")
        result = sandbox.run(
            node(workspace=ws), split="val", seed=0, timeout_s=60, data_dir=data,
            mem_limit_mb=1024, header_columns=("row_id", "listing_id", "prediction"),
        )
        assert result.ok, (result.error_class, result.error_excerpt)
        assert result.result_json["n_rows"] == 3
        rows = (ws / "submission.csv").read_text(encoding="utf-8").splitlines()
        assert rows[0] == "row_id,listing_id,prediction"
        assert [r.split(",")[1] for r in rows[1:]] == ["7", "8", "9"]


class TestDotenvParsing:
    """`.env` is how every real run gets its key and its provider, so a parsing quirk
    here fails the run and blames something else. This suite exists because
    `TECHJAM_LLM=openai   # primary provider` set the provider to the whole string and
    reported it as an unknown provider."""

    def _load(self, tmp_path, monkeypatch, body: str):
        env = tmp_path / ".env"
        env.write_text(body, encoding="utf-8")
        for var in ("TECHJAM_LLM", "TECHJAM_MODEL", "SOME_KEY"):
            monkeypatch.delenv(var, raising=False)
        agent_mod.load_dotenv(env)

    def test_inline_comment_is_stripped(self, tmp_path, monkeypatch):
        self._load(tmp_path, monkeypatch, "TECHJAM_LLM=openai   # primary provider\n")
        assert os.environ["TECHJAM_LLM"] == "openai"

    def test_a_hash_without_leading_space_is_part_of_the_value(self, tmp_path, monkeypatch):
        # Passwords and tokens legitimately contain '#'; only a whitespace-preceded
        # '#' starts a comment, which is what python-dotenv does.
        self._load(tmp_path, monkeypatch, "SOME_KEY=abc#def\n")
        assert os.environ["SOME_KEY"] == "abc#def"

    def test_quoted_values_keep_their_hash(self, tmp_path, monkeypatch):
        self._load(tmp_path, monkeypatch, 'SOME_KEY="abc # not a comment"\n')
        assert os.environ["SOME_KEY"] == "abc # not a comment"

    def test_existing_environment_still_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TECHJAM_MODEL", "already-set")
        env = tmp_path / ".env"
        env.write_text("TECHJAM_MODEL=from-file\n", encoding="utf-8")
        agent_mod.load_dotenv(env)
        assert os.environ["TECHJAM_MODEL"] == "already-set"

    def test_export_prefix_and_blank_lines(self, tmp_path, monkeypatch):
        self._load(tmp_path, monkeypatch, "\n# a comment\nexport TECHJAM_LLM=gemini\n\n")
        assert os.environ["TECHJAM_LLM"] == "gemini"
