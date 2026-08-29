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
    PIPELINE_CLI, Budget, Context, ExecResult, HistoryEntry, Idea, Node, TaskSpec,
)

_needs_openai = pytest.mark.skipif(
    importlib.util.find_spec("openai") is None,
    reason="optional provider; `pip install openai` to run these",
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
        ctx.history = [HistoryEntry(iteration=1, node_id="n000", kind="draft", hypothesis="try FM",
                             primary=None, delta_vs_baseline=None,
                                    status="buggy", error_class="import")]
        a = Agent(FakeClient([]))
        a.draft(ctx)
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
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("TECHJAM_LLM", raising=False)
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


class FakeOpenAISDK:
    """Stands in for openai.OpenAI(). Records what the adapter sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

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
        monkeypatch.delenv("TECHJAM_LLM", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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
