"""Fault-injection suite — B's headline Robustness deliverable.

Every row of the fault table in references/roles.md gets an end-to-end test: a real
subprocess, a real failure, and the classified ExecResult the orchestrator needs to
route around it. Nothing here may require human input, and nothing may leak a
process past the end of the test.

Results land in STATUS.md under `## Fault-injection results`; D quotes that table in
the writeup, so it has to be generated from real runs, not asserted from memory.
"""
from __future__ import annotations

import os
import subprocess
import time

import pytest

from orchestrator import sandbox

FAST = {"split": "val", "seed": 0, "timeout_s": 30, "mem_limit_mb": 1024}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


class TestHappyPath:
    def test_good_pipeline(self, make_node, data_dir):
        r = sandbox.run(make_node("good.py"), data_dir=data_dir, **FAST)
        assert r.ok and r.exit_code == 0
        assert r.error_class is None and r.error_excerpt is None
        assert r.result_json["n_rows"] == 20
        assert r.artifacts["submission"].is_file()
        assert r.wall_s > 0 and r.peak_rss_mb >= 0

    def test_subsample_is_passed_through(self, make_node, data_dir):
        node = make_node("good.py")
        r = sandbox.run(node, subsample=0.5, data_dir=data_dir, **FAST)
        assert r.ok and r.result_json["n_rows"] == 10

    def test_logs_are_written_to_the_workspace(self, make_node, data_dir):
        node = make_node("good.py")
        sandbox.run(node, data_dir=data_dir, **FAST)
        assert (node.workspace / "stdout.log").is_file()
        assert (node.workspace / "stderr.log").is_file()

    def test_stale_submission_is_cleared_before_a_rerun(self, make_node, data_dir):
        """A repair that fails must not inherit its parent's submission and score
        like a success."""
        node = make_node("no_submission.py")
        (node.workspace / "submission.csv").write_text(
            f"{sandbox.SUBMISSION_HEADER}\n0,u1,v1,0.9\n")
        r = sandbox.run(node, data_dir=data_dir, **FAST)
        assert not r.ok and r.error_class == "contract"
        assert not (node.workspace / "submission.csv").exists()


class TestFaultTable:
    """One test per row of the fault table. Each asserts the class the repair loop
    keys on, and that the excerpt is small enough to put in a prompt."""

    @pytest.mark.parametrize("fixture,expected", [
        ("syntax_error.py", "syntax"),
        ("bad_import.py", "import"),
        ("runtime_error.py", "runtime"),
        ("no_submission.py", "contract"),
        ("no_result_json.py", "contract"),
        ("bad_header.py", "contract"),
        ("row_mismatch.py", "contract"),
        ("nan_scores.py", "eval"),
    ])
    def test_class_and_excerpt(self, make_node, data_dir, fixture, expected):
        r = sandbox.run(make_node(fixture), data_dir=data_dir, **FAST)
        assert not r.ok
        assert r.error_class == expected, (fixture, r.error_excerpt, r.stderr_tail)
        assert r.error_excerpt and len(r.error_excerpt) <= sandbox.EXCERPT_CHARS
        assert not r.artifacts

    def test_runtime_excerpt_names_the_failing_line_not_the_whole_stack(
            self, make_node, data_dir):
        r = sandbox.run(make_node("runtime_error.py"), data_dir=data_dir, **FAST)
        assert "KeyError" in r.error_excerpt
        assert "level_three" in r.error_excerpt      # deepest pipeline frame
        assert "level_one" not in r.error_excerpt    # shallower frames dropped
        assert len(r.error_excerpt.splitlines()) <= 5

    def test_infinite_loop_times_out(self, make_node, data_dir):
        started = time.monotonic()
        r = sandbox.run(make_node("infinite_loop.py"), split="val", seed=0,
                        timeout_s=2, data_dir=data_dir, mem_limit_mb=1024)
        assert not r.ok and r.error_class == "timeout"
        assert time.monotonic() - started < 20      # killed promptly, not waited out
        assert "training" in r.stdout_tail          # partial output still captured

    def test_timeout_kills_the_whole_process_group(self, make_node, data_dir):
        """An orphaned grandchild would eat a core for the rest of a six-hour run."""
        node = make_node("orphan_spawner.py")
        r = sandbox.run(node, split="val", seed=0, timeout_s=2,
                        data_dir=data_dir, mem_limit_mb=1024)
        assert r.error_class == "timeout"
        pid = int(r.stdout_tail.split("SPAWNED ")[1].split()[0])
        deadline = time.monotonic() + 5
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not alive(pid), f"grandchild {pid} outlived its process group"

    def test_memory_hog_is_killed_and_classified_oom(self, make_node, data_dir):
        r = sandbox.run(make_node("memory_hog.py"), split="val", seed=0,
                        timeout_s=60, mem_limit_mb=384, data_dir=data_dir)
        assert not r.ok
        assert r.error_class == "oom", (r.error_excerpt, r.stderr_tail)
        assert r.peak_rss_mb > 0

    def test_network_access_is_blocked(self, make_node, data_dir):
        """Downloading external data would breach the one disqualifying rule."""
        r = sandbox.run(make_node("network.py"), data_dir=data_dir, **FAST)
        assert not r.ok and r.error_class == "data"
        assert "network access is disabled" in r.stderr_tail

    def test_a_pipeline_that_prompts_does_not_hang(self, make_node, data_dir):
        r = sandbox.run(make_node("stdin_reader.py"), data_dir=data_dir, **FAST)
        assert not r.ok and r.error_class == "runtime"
        assert "EOFError" in r.stderr_tail


class TestIsolation:
    def test_no_secret_reaches_the_child(self, make_node, data_dir, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-appear")
        monkeypatch.setenv("SOME_OTHER_TOKEN", "nope")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "nope")
        node = make_node("env_probe.py")
        r = sandbox.run(node, data_dir=data_dir, **FAST)
        assert r.ok and r.result_json["leaked"] == []
        for path in node.workspace.rglob("*"):
            if path.is_file():
                assert "sk-ant-should-never-appear" not in path.read_text(errors="replace")

    def test_seed_reaches_the_child_as_pythonhashseed(self, make_node, data_dir):
        r = sandbox.run(make_node("env_probe.py"), split="val", seed=7,
                        timeout_s=30, mem_limit_mb=1024, data_dir=data_dir)
        assert r.result_json["hashseed"] == "7"

    def test_missing_pipeline_is_an_orchestrator_bug_not_an_exec_result(
            self, make_node, data_dir):
        node = make_node("good.py")
        (node.workspace / "pipeline.py").unlink()
        with pytest.raises(FileNotFoundError):
            sandbox.run(node, data_dir=data_dir, **FAST)


def test_no_python_processes_leak_across_the_suite():
    """Backstop for the whole file: nothing named after our fixtures survives.

    Allows a short grace period so a SIGKILL still being reaped does not read as a
    leak, but the process must be gone — a stray trainer eats a core for the rest
    of a six-hour run."""
    deadline = time.monotonic() + 3
    while True:
        out = subprocess.run(["ps", "-A", "-o", "command="],
                             capture_output=True, text=True).stdout
        stragglers = [ln for ln in out.splitlines()
                      if "pipeline.py" in ln and "--out-dir" in ln]
        if not stragglers or time.monotonic() > deadline:
            break
        time.sleep(0.25)
    assert not stragglers, stragglers


# --------------------------------------------------------------------------- #
# repair loop: the sandbox and the agent composed
# --------------------------------------------------------------------------- #

BROKEN = "import argparse\ndef broken(:\n    return 1\n"

FIXED = '''\
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--data-dir"); p.add_argument("--out-dir")
p.add_argument("--split"); p.add_argument("--seed", type=int)
p.add_argument("--subsample", type=float, default=None)
a = p.parse_args()
with open(a.out_dir + "/submission.csv", "w") as fh:
    fh.write("row_id,user_id,video_id,score\\n0,u1,v1,0.9\\n")
print("RESULT_JSON " + json.dumps({"n_rows": 1, "train_seconds": 0.0, "notes": "fixed"}))
'''


class ScriptedAgentClient:
    """Returns each code string in turn, wrapped as a tool call."""

    def __init__(self, codes):
        self.codes = list(codes)
        self.prompts = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        code = self.codes.pop(0) if self.codes else FIXED
        payload = {"hypothesis": "h", "plan": ["p"], "code": code, "idea_ids": []}
        block = type("B", (), {"type": "tool_use", "name": "submit_pipeline",
                               "input": payload})()
        usage = type("U", (), {"input_tokens": 10, "output_tokens": 5,
                               "cache_creation_input_tokens": 0,
                               "cache_read_input_tokens": 0})()
        return type("M", (), {"content": [block], "usage": usage})()


def drive_repairs(node, agent, ctx, data_dir, max_attempts=None):
    """Stand-in for A's loop, so the agent/sandbox seam is exercised end to end.
    Returns (final ExecResult, attempts used)."""
    from orchestrator.agent import repair_exhausted

    result = sandbox.run(node, split="val", seed=0, timeout_s=30,
                         data_dir=data_dir, mem_limit_mb=1024)
    node.exec_result = result
    while not result.ok and not repair_exhausted(node):
        if max_attempts is not None and node.repair_attempts >= max_attempts:
            break
        proposal = agent.repair(ctx, node)
        node.repair_attempts += 1
        (node.workspace / "pipeline.py").write_text(proposal.code)
        result = sandbox.run(node, split="val", seed=0, timeout_s=30,
                             data_dir=data_dir, mem_limit_mb=1024)
        node.exec_result = result
    node.status = "ok" if result.ok else ("dead" if repair_exhausted(node) else "buggy")
    return result, node.repair_attempts


@pytest.fixture
def repair_ctx():
    from pathlib import Path as _Path

    from orchestrator.contracts import Context, TaskSpec
    task = TaskSpec(name="kuairand-pure", data_dir=_Path("data"),
                    metrics=("gauc", "ndcg@5"),
                    baseline_val={"primary": 0.6016}, baseline_test={"primary": 0.5946})
    return Context(task=task, data_card="tiny", run_id="r-repair", iteration=1)


class TestRepairLoop:
    def test_a_syntax_error_is_repaired_and_the_run_continues(
            self, make_node, data_dir, repair_ctx):
        from orchestrator.agent import Agent
        node = make_node("good.py")
        (node.workspace / "pipeline.py").write_text(BROKEN)
        agent = Agent(ScriptedAgentClient([FIXED]))
        result, attempts = drive_repairs(node, agent, repair_ctx, data_dir)
        assert result.ok and attempts == 1 and node.status == "ok"

    def test_each_attempt_sees_the_error_and_the_previous_attempts(
            self, make_node, data_dir, repair_ctx):
        from orchestrator.agent import Agent
        node = make_node("good.py")
        (node.workspace / "pipeline.py").write_text(BROKEN)
        client = ScriptedAgentClient([BROKEN, BROKEN, FIXED])
        result, attempts = drive_repairs(node, Agent(client), repair_ctx, data_dir)
        assert result.ok and attempts == 3
        assert "SyntaxError" in client.prompts[0]
        assert "first repair attempt" in client.prompts[0]
        assert "1 previous repair attempt(s) failed" in client.prompts[1]
        assert "2 previous repair attempt(s) failed" in client.prompts[2]

    def test_after_three_failures_the_node_is_dead_not_looping(
            self, make_node, data_dir, repair_ctx):
        from orchestrator.agent import MAX_REPAIR_ATTEMPTS, Agent
        node = make_node("good.py")
        (node.workspace / "pipeline.py").write_text(BROKEN)
        client = ScriptedAgentClient([BROKEN] * 10)
        result, attempts = drive_repairs(node, Agent(client), repair_ctx, data_dir)
        assert not result.ok
        assert attempts == MAX_REPAIR_ATTEMPTS
        assert node.status == "dead"                 # A routes around it from here

    def test_an_import_fault_is_repaired_to_an_allowed_library(
            self, make_node, data_dir, repair_ctx):
        from orchestrator.agent import Agent
        node = make_node("bad_import.py")
        client = ScriptedAgentClient([FIXED])
        result, attempts = drive_repairs(node, Agent(client), repair_ctx, data_dir)
        assert result.ok and attempts == 1
        assert "definitely_not_a_real_library_xyz" in client.prompts[0]

    def test_a_timeout_is_repaired_without_waiting_out_the_clock(
            self, make_node, data_dir, repair_ctx):
        from orchestrator.agent import Agent
        node = make_node("infinite_loop.py")
        started = time.monotonic()
        result = sandbox.run(node, split="val", seed=0, timeout_s=2,
                             data_dir=data_dir, mem_limit_mb=1024)
        node.exec_result = result
        assert result.error_class == "timeout"
        agent = Agent(ScriptedAgentClient([FIXED]))
        proposal = agent.repair(repair_ctx, node)
        (node.workspace / "pipeline.py").write_text(proposal.code)
        after = sandbox.run(node, split="val", seed=0, timeout_s=30,
                            data_dir=data_dir, mem_limit_mb=1024)
        assert after.ok
        assert time.monotonic() - started < 30
