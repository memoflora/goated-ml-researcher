"""Loop-level acceptance tests. Owner: A.

Each test here maps to a line in the "Done when" list for role A in roles.md.
Most of it runs against the stubs and is near-instant. The end-to-end block at the
bottom is the exception: it drives the real sandbox, evaluator and dataset, and is
gated on the data being present (plus TECHJAM_SLOW_TESTS=1 for the two that train).
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchestrator import journal as journal_mod
from orchestrator.core import Orchestrator, atomic_write, rank_average
from tests.stubs import StubAgent, StubEvaluator, StubExecutor, stub_task
from tests.stubs.executor import CLIMBING_SCRIPT, DEFAULT_SCRIPT

REPO_ROOT = Path(__file__).resolve().parents[1]


def build(tmp_path, *, script=DEFAULT_SCRIPT, max_iters=3, run_id="rtest", **kwargs):
    run_dir = Path(tmp_path) / run_id
    journal_mod.close()
    return Orchestrator(
        stub_task(max_iters=max_iters, wall_clock_s=3600),
        run_dir=run_dir,
        run_id=run_id,
        agent=StubAgent(),
        executor=StubExecutor(script),
        evaluator=StubEvaluator(),
        mode="smoke",
        journal=journal_mod.Journal(run_dir / "journal.jsonl", run_id, fsync=False),
        **kwargs,
    )


def rows(run_dir):
    return list(journal_mod.read(Path(run_dir) / "journal.jsonl"))


def events(run_dir, name):
    return [r for r in rows(run_dir) if r.get("event") == name]


# -- the smoke run --------------------------------------------------------


def test_smoke_run_three_iterations_end_to_end(tmp_path):
    orch = build(tmp_path)
    summary = orch.run()

    assert summary["iterations"] == 3
    assert summary["best_node"] is not None
    assert summary["final_valid"] is True
    assert Path(summary["final_submission"]).exists()
    assert (orch.run_dir / "best" / "pipeline.py").exists()
    assert (orch.run_dir / "interventions.md").exists()
    # Every node keeps the exact program that produced its score.
    for node in orch.tree.nodes.values():
        assert (node.workspace / "pipeline.py").read_text(encoding="utf-8") == node.proposal.code


def test_journal_is_valid_jsonl_and_carries_the_graded_fields(tmp_path):
    orch = build(tmp_path)
    orch.run()
    raw = (orch.run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert raw and all(json.loads(line) for line in raw)

    seen = {r["event"] for r in rows(orch.run_dir)}
    assert {"run_start", "proposal", "exec", "eval", "run_end"} <= seen
    for proposal in events(orch.run_dir, "proposal"):
        assert proposal["hypothesis"].strip(), "Innovation is scored on this field"
        assert proposal["reason"], "a judge must be able to read why the agent moved here"
        assert proposal["model"] and "tokens_in" in proposal
    for row in rows(orch.run_dir):
        assert set(row) >= {"ts", "run_id", "event"}


def test_run_start_is_emitted_exactly_once(tmp_path):
    orch = build(tmp_path)
    orch.run()
    assert len(events(orch.run_dir, "run_start")) == 1


def test_config_and_state_are_written(tmp_path):
    orch = build(tmp_path)
    orch.run()
    cfg = json.loads((orch.run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["task"]["conv_eps"] == 0.002 and cfg["git_sha"]
    state = json.loads((orch.run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["tree"]["nodes"] and state["stop_reason"] == "max_iters"


# -- robustness -----------------------------------------------------------


def test_failed_node_is_repaired_not_fatal(tmp_path):
    orch = build(tmp_path, script=("err:syntax", "ok:0.6100"), max_iters=2)
    orch.run()
    assert orch.tree.nodes["n000"].status == "buggy"
    assert orch.tree.nodes["n001"].kind == "debug"
    assert orch.tree.nodes["n001"].parent_id == "n000"
    assert orch.tree.nodes["n001"].status == "ok"
    assert events(orch.run_dir, "recovery")[0]["recovery"] == "schedule_debug"


def test_three_failures_kill_the_node_and_the_run_routes_around_it(tmp_path):
    orch = build(tmp_path, script=("err:syntax",) * 4 + ("ok:0.6100",) * 2, max_iters=6)
    summary = orch.run()

    chain = [orch.tree.nodes[f"n{i:03d}"] for i in range(4)]
    assert [n.repair_attempts for n in chain] == [0, 1, 2, 3]
    assert chain[-1].status == "dead"
    prune = events(orch.run_dir, "prune")
    assert prune and prune[0]["recovery"] == "route_around"
    # ...and the run keeps going and still produces a submission.
    assert summary["iterations"] == 6
    assert summary["best_node"] is not None
    assert summary["dead_nodes"] == 1


def test_a_dead_run_still_finishes_cleanly(tmp_path):
    orch = build(tmp_path, script=("err:oom",) * 12, max_iters=8)
    summary = orch.run()
    assert summary["best_node"] is None
    assert summary["final_submission"] is None
    assert events(orch.run_dir, "run_end")  # no crash, no hang, no human


def test_a_buggy_node_never_becomes_best(tmp_path):
    orch = build(tmp_path, script=("ok:0.6100", "err:eval", "ok:0.6050"), max_iters=3)
    orch.run()
    assert orch.best_id == "n000"


def test_validation_failure_is_treated_as_a_contract_error(tmp_path):
    class RejectingEvaluator(StubEvaluator):
        def validate(self, submission, split="val"):
            return False, "row_id gap at 12"

    orch = build(tmp_path, script=("ok:0.6100", "ok:0.6100"), max_iters=2)
    orch.evaluator = RejectingEvaluator()
    orch.run()
    assert orch.tree.nodes["n000"].status == "buggy"
    err = events(orch.run_dir, "error")[0]
    assert err["error_class"] == "contract" and "row_id gap" in err["error_excerpt"]


def test_an_agent_exception_costs_an_iteration_not_the_run(tmp_path):
    class FlakyAgent(StubAgent):
        def draft(self, ctx):
            if not self.calls:
                self.calls.append("boom")
                raise RuntimeError("connection reset by peer")
            return super().draft(ctx)

    orch = build(tmp_path, script=("ok:0.6100",) * 3, max_iters=3)
    orch.agent = FlakyAgent()
    summary = orch.run()
    assert summary["best_node"] is not None
    assert any(e["error_class"] == "unknown" for e in events(orch.run_dir, "error"))


# -- convergence and search ----------------------------------------------


def test_run_stops_on_convergence_not_on_the_iteration_cap(tmp_path):
    orch = build(tmp_path, script=DEFAULT_SCRIPT, max_iters=50)
    summary = orch.run()
    assert summary["stop_reason"] == "converged"
    assert summary["iterations"] == len(DEFAULT_SCRIPT)
    assert events(orch.run_dir, "converged")
    # errored iterations never entered the convergence window
    assert len(orch.best_history) == sum(1 for s in DEFAULT_SCRIPT if s.startswith("ok:"))


def test_errors_alone_never_declare_convergence(tmp_path):
    orch = build(tmp_path, script=("err:runtime",) * 20, max_iters=6)
    summary = orch.run()
    assert summary["stop_reason"] == "max_iters"
    assert orch.best_history == []


def test_three_flat_iterations_trigger_an_explore(tmp_path):
    script = ("ok:0.6000", "ok:0.6015", "ok:0.6030", "ok:0.6045", "ok:0.6060")
    orch = build(tmp_path, script=script, max_iters=5)
    orch.run()
    assert orch.flat_iters >= 3
    reasons = [p["reason"] for p in events(orch.run_dir, "proposal")]
    assert any("exploring the second-best" in r for r in reasons)


def test_fifty_iteration_run_completes_with_a_valid_journal(tmp_path):
    orch = build(tmp_path, script=CLIMBING_SCRIPT, max_iters=50)
    summary = orch.run()
    assert summary["iterations"] == 50 and summary["stop_reason"] == "max_iters"
    lines = (orch.run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line)["run_id"] == "rtest" for line in lines)
    assert len(orch.tree.nodes) == 50


def test_budget_guard_refuses_an_iteration_that_cannot_finish(tmp_path):
    orch = build(tmp_path, script=CLIMBING_SCRIPT, max_iters=50)
    orch.task = stub_task(max_iters=50, wall_clock_s=1)  # one second of ceiling
    summary = orch.run()
    assert summary["stop_reason"] == "wall_clock" and summary["iterations"] == 0


def test_token_budget_stops_the_run(tmp_path):
    orch = build(tmp_path, script=CLIMBING_SCRIPT, max_iters=50)
    orch.acct.token_budget = 3000
    summary = orch.run()
    assert summary["stop_reason"] == "token_budget"
    assert summary["tokens_total"] >= 3000


# -- accounting -----------------------------------------------------------


def test_accounting_matches_the_journal_exactly(tmp_path):
    orch = build(tmp_path, script=CLIMBING_SCRIPT, max_iters=7)
    summary = orch.run()
    proposals = events(orch.run_dir, "proposal")
    assert summary["tokens_in"] == sum(p["tokens_in"] for p in proposals)
    assert summary["tokens_out"] == sum(p["tokens_out"] for p in proposals)
    assert summary["llm_calls"] == len(proposals) == 7
    assert summary["iterations"] == 7


# -- context --------------------------------------------------------------


def test_context_never_carries_past_code(tmp_path):
    orch = build(tmp_path, script=CLIMBING_SCRIPT, max_iters=5)
    orch.run()
    history = orch.compact_history()
    assert history and all(len(h.hypothesis) <= 260 for h in history)
    assert all(not hasattr(h, "code") for h in history)


def test_improve_context_carries_exactly_one_parent_program(tmp_path):
    from orchestrator import policy

    orch = build(tmp_path, script=CLIMBING_SCRIPT, max_iters=4)
    orch.run()
    action = policy.next_action(orch.tree.nodes, flat_iters=0)
    parent = orch.tree.get(action.parent_id)
    ctx = orch.build_context(action, parent)
    assert ctx.parent_code == parent.proposal.code
    assert ctx.parent_metrics == parent.metrics
    assert ctx.error_excerpt is None


def test_repair_context_carries_the_error_and_what_was_already_tried(tmp_path):
    from orchestrator import policy

    orch = build(tmp_path, script=("err:import", "err:import"), max_iters=2)
    orch.run()
    action = policy.next_action(orch.tree.nodes)
    target = orch.tree.get(action.parent_id)
    ctx = orch.build_context(action, target, repair_attempts=target.repair_attempts + 1)
    assert action.kind == "debug"
    assert ctx.error_class == "import" and "ModuleNotFoundError" in ctx.error_excerpt
    assert ctx.prior_repair_plans, "attempt 3 must not repeat attempt 1's context"
    assert ctx.repair_attempt == 2


# -- finalisation ---------------------------------------------------------


def test_final_submission_is_the_best_node_rerun_on_the_test_split(tmp_path):
    orch = build(tmp_path, script=CLIMBING_SCRIPT, max_iters=4, final_seeds=(0, 1, 2))
    summary = orch.run()
    final = Path(summary["final_submission"])
    assert final.name == "submission.csv" and final.parent.name == "final"
    assert final.read_text(encoding="utf-8").splitlines()[0] == "row_id,user_id,video_id,score"
    for seed in (0, 1, 2):
        assert (orch.run_dir / "final" / f"seed{seed}" / "pipeline.py").exists()
    best_code = orch.tree.nodes[summary["best_node"]].proposal.code
    assert (orch.run_dir / "final" / "seed1" / "pipeline.py").read_text(encoding="utf-8") == best_code


def test_finalisation_survives_a_failing_seed(tmp_path):
    class OneBadSeed(StubExecutor):
        def run(self, node, *, split="val", seed=0, **kw):
            if split == "test" and seed == 1:
                self.script.insert(self.calls, "err:runtime")
            return super().run(node, split=split, seed=seed, **kw)

    orch = build(tmp_path, script=list(CLIMBING_SCRIPT), max_iters=3, final_seeds=(0, 1, 2))
    orch.executor = OneBadSeed(list(CLIMBING_SCRIPT))
    summary = orch.run()
    assert summary["final_valid"] is True  # two good seeds still make a submission


def test_rank_average_orders_by_mean_rank(tmp_path):
    def write(path, scores):
        path.write_text(
            "row_id,user_id,video_id,score\n"
            + "".join(f"{i},{i},{i},{s}\n" for i, s in enumerate(scores)),
            encoding="utf-8",
        )

    a, b, out = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "out.csv"
    write(a, [0.1, 0.9, 0.5])
    write(b, [10.0, 30.0, 20.0])  # same order, wildly different scale
    rank_average([a, b], out)
    lines = out.read_text(encoding="utf-8").splitlines()[1:]
    scores = [float(line.split(",")[3]) for line in lines]
    assert scores[1] > scores[2] > scores[0]


# -- checkpoint and resume ------------------------------------------------


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "state.json"
    atomic_write(target, '{"a": 1}')
    atomic_write(target, '{"a": 2}')
    assert json.loads(target.read_text(encoding="utf-8"))["a"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_rebuilds_the_tree_and_continues(tmp_path):
    first = build(tmp_path, script=CLIMBING_SCRIPT, max_iters=4, run_id="rres")
    first.run()
    node_ids = set(first.tree.nodes)
    journal_mod.close()

    second = Orchestrator.resume(
        first.run_dir,
        "rres",
        agent=StubAgent(),
        executor=StubExecutor(CLIMBING_SCRIPT[4:]),
        evaluator=StubEvaluator(),
        task=stub_task(max_iters=8, wall_clock_s=3600),
        journal=journal_mod.Journal(first.run_dir / "journal.jsonl", "rres", fsync=False),
    )
    assert set(second.tree.nodes) == node_ids
    assert second.iteration == 4 and second.best_id == first.best_id
    assert second.acct.tokens_in == first.acct.tokens_in

    summary = second.run()
    assert summary["iterations"] == 8
    assert node_ids < set(second.tree.nodes), "resumed run must extend the same tree"
    assert any(r.get("recovery") == "resumed" for r in rows(first.run_dir))


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL")
def test_kill_dash_nine_then_resume_loses_no_checkpointed_node(tmp_path):
    """The run that matters is the six-hour one. It must survive being killed."""
    run_dir = tmp_path / "rkill"
    child = tmp_path / "child.py"
    child.write_text(
        textwrap.dedent(
            f"""
            import os, sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from orchestrator.core import Orchestrator
            from tests.stubs import StubAgent, StubEvaluator, StubExecutor, stub_task
            from tests.stubs.executor import CLIMBING_SCRIPT

            class Killer(StubExecutor):
                def run(self, node, **kw):
                    if self.calls >= 3:          # die inside iteration 4
                        os.kill(os.getpid(), 9)
                    return super().run(node, **kw)

            Orchestrator(
                stub_task(max_iters=10, wall_clock_s=600),
                run_dir={str(run_dir)!r}, run_id="rkill",
                agent=StubAgent(), executor=Killer(CLIMBING_SCRIPT),
                evaluator=StubEvaluator(), mode="smoke",
            ).run()
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run([sys.executable, str(child)], capture_output=True, timeout=60, check=False)
    assert proc.returncode == -9, "the child was supposed to be SIGKILLed"

    # Nothing on disk is torn: the checkpoint parses and the journal still reads.
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["iteration"] == 3 and len(state["tree"]["nodes"]) == 3
    assert all(json.loads(line) for line in (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines())

    journal_mod.close()
    resumed = Orchestrator.resume(
        run_dir,
        "rkill",
        agent=StubAgent(),
        executor=StubExecutor(CLIMBING_SCRIPT[3:]),
        evaluator=StubEvaluator(),
        task=stub_task(max_iters=6, wall_clock_s=600),
        journal=journal_mod.Journal(run_dir / "journal.jsonl", "rkill", fsync=False),
    )
    assert set(resumed.tree.nodes) == {"n000", "n001", "n002"}
    summary = resumed.run()
    assert summary["iterations"] == 6 and summary["best_node"] is not None


def test_two_runs_in_the_same_minute_never_share_a_directory(tmp_path):
    """A second run must not append to the first run's graded journal."""
    from datetime import datetime, timezone

    from orchestrator.core import new_run_id

    now = datetime(2026, 8, 30, 4, 12, tzinfo=timezone.utc)
    first = new_run_id(now, runs_dir=tmp_path)
    (tmp_path / first).mkdir()
    second = new_run_id(now, runs_dir=tmp_path)
    (tmp_path / second).mkdir()
    third = new_run_id(now, runs_dir=tmp_path)
    assert first == "r20260830-0412"
    assert len({first, second, third}) == 3


def test_atomic_write_survives_a_filesystem_that_refuses_directory_fsync(tmp_path, monkeypatch):
    """Windows cannot open a directory as a fd, and some filesystems refuse it.

    os.replace is already atomic everywhere; the directory fsync is a POSIX
    durability bonus and must never be able to fail a checkpoint.
    """
    from orchestrator import core

    real_open = os.open

    def refusing_open(path, flags, *args, **kwargs):
        if flags == os.O_RDONLY and Path(path).is_dir():
            raise PermissionError(13, "Permission denied")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(core.os, "open", refusing_open)
    target = tmp_path / "state.json"
    atomic_write(target, '{"iteration": 7}')
    assert json.loads(target.read_text(encoding="utf-8"))["iteration"] == 7
    assert not list(tmp_path.glob("*.tmp"))


def test_runtime_imports_stay_within_the_pinned_python_version(tmp_path):
    """`typing.Self` is 3.11+. Importing it at runtime broke a teammate's setup.

    Type-checker-only imports under `if TYPE_CHECKING:` are fine; runtime ones
    are not, because they fail at import time and take the whole run with them.
    """
    import ast

    offenders = []
    for path in sorted(Path(REPO_ROOT / "orchestrator").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded = {
            id(child)
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and "TYPE_CHECKING" in ast.dump(node.test)
            for child in ast.walk(node)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                if id(node) in guarded:
                    continue
                offenders += [
                    f"{path.name}: {alias.name}"
                    for alias in node.names
                    if alias.name in {"Self", "Never", "LiteralString", "TypeVarTuple"}
                ]
    assert not offenders, f"3.11+ typing names imported at runtime: {offenders}"


def test_journal_round_trips_non_ascii_whatever_the_locale(tmp_path):
    """A real LLM writes em dashes and ellipses. Windows reads cp1252 by default.

    Writing was always explicit utf-8; reading back without an encoding was not,
    and that combination is a UnicodeDecodeError on any non-utf-8 locale.
    """
    hypothesis = "FM ignores duration bias — long videos dominate…"
    j = journal_mod.Journal(tmp_path / "journal.jsonl", "r1", fsync=False)
    j.emit({"event": "proposal", "hypothesis": hypothesis})
    j.close()
    assert [r["hypothesis"] for r in journal_mod.read(tmp_path / "journal.jsonl")] == [hypothesis]


def test_no_text_file_io_relies_on_the_platform_default_encoding():
    """Windows defaults to cp1252, Linux/macOS to utf-8. Never rely on either.

    Scoped to the files A owns. Offenders elsewhere are reported to their owner
    in STATUS.md rather than failing their build; `make check` staying green is
    the team's top priority, and this is not my lane to enforce in.
    """
    import ast

    owned = [REPO_ROOT / "orchestrator" / f"{m}.py" for m in
             ("contracts", "core", "policy", "journal", "run")]
    owned += [REPO_ROOT / "tests" / "test_core.py", REPO_ROOT / "tests" / "test_policy.py"]
    owned += sorted((REPO_ROOT / "tests" / "stubs").glob("*.py"))

    offenders = []
    for path in owned:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            # bare open(...) is the dangerous form; .read_text()/.write_text()/.open() too
            if isinstance(node.func, ast.Name):
                if node.func.id != "open":
                    continue
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in {"read_text", "write_text", "open"}:
                    continue
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    continue  # os.open takes fd flags, not an encoding
            else:
                continue
            if any(kw.arg == "encoding" for kw in node.keywords):
                continue
            if any(isinstance(a, ast.Constant) and "b" in str(a.value) for a in node.args):
                continue  # binary mode
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"text I/O without an explicit encoding: {offenders}"


# --------------------------------------------------------------------- end to end
#
# The proof that had been missing. Every other test in this file drives the loop with
# stubs; these two drive it with the REAL sandbox, the REAL evaluator and the REAL
# dataset, and assert that a run actually lands a scored node and a valid final
# submission. Before this existed the loop had never once produced a scored node on
# real data, and nothing in the suite would have noticed.
#
# The agent is scripted rather than live on purpose: a test that needs an API key is a
# test CI cannot run, and we want the harness verified on every push independently of
# whether any model is reachable or any key has budget.

_REPO = Path(__file__).resolve().parent.parent
_KUAIRAND = _REPO / "data" / "KuaiRand-Pure" / "data"
_POP_FIXTURE = _REPO / "tests" / "fixtures" / "pipelines" / "kuairand_pop.py"

_needs_dataset = pytest.mark.skipif(
    not (_KUAIRAND / "log_standard_4_22_to_5_08_pure.csv").is_file(),
    reason="needs the real KuaiRand-Pure dataset (gitignored); see README quickstart",
)
_slow = pytest.mark.skipif(
    os.environ.get("TECHJAM_SLOW_TESTS") != "1",
    reason="real training run; set TECHJAM_SLOW_TESTS=1 to include",
)

# Item popularity on validation, measured through this exact stack. The organisers
# publish 0.5807; we reproduce it to five decimals.
_POP_VAL_PRIMARY = 0.58072
_TOL = 0.0008  # the baseline's own five-seed std


class _ScriptedAgent:
    """Serves a real pipeline as if a model had written it.

    Deliberately not a mock of the Agent class: it implements the same three methods
    the loop actually calls, so if that seam changes shape this test fails rather than
    passing against an interface nobody uses any more.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.calls: list[str] = []

    def _proposal(self, kind: str):
        from orchestrator.contracts import Proposal

        self.calls.append(kind)
        return Proposal(
            hypothesis=(
                "Item popularity is a strong prior on this data and needs no fitting "
                "beyond counting, so it establishes a floor before anything clever."
            ),
            plan=["count positives per item", "smooth toward the global rate"],
            code=self.source,
            idea_ids=["T0.item-pop-prior"],
            tokens_in=1200,
            tokens_out=300,
            model="scripted",
        )

    def draft(self, ctx):
        return self._proposal("draft")

    def improve(self, ctx, parent):
        return self._proposal("improve")

    def repair(self, ctx, node):
        return self._proposal("repair")


def _run_real(tmp_path, *, max_iters=2, final_seeds=(0,)):
    from orchestrator import evaluate as evaluate_mod
    from orchestrator import sandbox as sandbox_mod
    from orchestrator.core import Orchestrator
    from orchestrator.journal import Journal
    from orchestrator.run import MODES, build_task, parse_args

    # Build the TaskSpec through run.py's own parser and builder rather than
    # hand-rolling one. Two reasons: load_task() returns a TaskConfig, which is a
    # different type from the TaskSpec the loop consumes; and going through parse_args
    # means this test keeps tracking the CLI instead of drifting from it.
    args = parse_args(
        ["--task", "kuairand-pure", "--mode", "smoke", "--max-iters", str(max_iters)]
    )
    task = build_task(args, MODES["smoke"])
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    agent = _ScriptedAgent(_POP_FIXTURE.read_text(encoding="utf-8"))

    with Journal(run_dir / "journal.jsonl", "r-e2e") as journal:
        orch = Orchestrator(
            task,
            run_dir=run_dir,
            run_id="r-e2e",
            agent=agent,
            executor=sandbox_mod,
            evaluator=evaluate_mod,
            mode="smoke",
            # smoke mode subsamples to 2% of users, which is right for CI and wrong
            # here: we are asserting an exact published score, so the fit has to see
            # the same data the published number saw. 1.0 is "all users".
            subsample=1.0,
            n_drafts=1,
            final_seeds=final_seeds,
            journal=journal,
        )
        summary = orch.run()
    return summary, run_dir, agent


@_needs_dataset
@_slow
def test_the_loop_scores_a_real_pipeline_on_real_data(tmp_path):
    """A scored node, with the number we independently know to be correct.

    This is the regression guard for the bug that motivated the test: the stub
    pipeline emitted fabricated (user_id, video_id) pairs, so every submission failed
    alignment and `best_node` was always None. The suite was green throughout.
    """
    summary, run_dir, agent = _run_real(tmp_path, max_iters=1)

    assert summary["best_node"] is not None, (
        "the loop produced no scored node on real data; journal at "
        f"{run_dir / 'journal.jsonl'}"
    )
    assert agent.calls, "the loop never asked the agent for a proposal"

    primary = summary["best_metrics"]["primary"]
    assert abs(primary - _POP_VAL_PRIMARY) < _TOL, (
        f"item popularity scored {primary:.5f} through the loop, but scores "
        f"{_POP_VAL_PRIMARY} when run directly — the loop is altering the result"
    )


@_needs_dataset
@_slow
def test_the_run_lands_a_final_submission_that_validates(tmp_path):
    """Finalisation is the last thing that happens and the only thing we submit.

    It reruns the validation-best node on --split test, so it exercises a code path
    no development iteration ever touches. A run that scores well and then writes an
    unusable CSV is worth nothing.
    """
    summary, run_dir, _ = _run_real(tmp_path, max_iters=1)

    final = summary.get("final_submission")
    assert final, "run finished without writing a final submission"
    final_path = Path(final)
    assert final_path.is_file()
    assert summary.get("final_valid") is True, "the final submission did not validate"

    from orchestrator.evaluate import validate
    from orchestrator.taskspec import load_task

    ok, msg = validate(final_path, "test", load_task("kuairand-pure"))
    assert ok, f"final submission rejected on re-check: {msg}"

    header = final_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "row_id,user_id,video_id,score", f"wrong header: {header}"


@_needs_dataset
def test_the_pop_fixture_is_wired_to_the_loop_correctly(tmp_path):
    """Fast guard, runs on every push.

    Proves the scripted agent hands the loop a pipeline the sandbox can execute and
    the validator accepts. It stops one iteration short of scoring, so it costs
    seconds rather than minutes while still failing if the wiring rots.
    """
    from orchestrator import sandbox as sandbox_mod
    from orchestrator.contracts import Node
    from orchestrator.evaluate import validate
    from orchestrator.taskspec import load_task

    ws = tmp_path / "n000"
    ws.mkdir(parents=True)
    shutil.copy(_POP_FIXTURE, ws / "pipeline.py")
    node = Node(id="n000", parent_id=None, kind="draft", iteration=0, workspace=ws)

    res = sandbox_mod.run(node, split="val", seed=0, timeout_s=600, data_dir=_KUAIRAND)
    assert res.ok, f"pipeline failed in the sandbox: {(res.stderr_tail or '')[-400:]}"
    sub = res.artifacts.get("submission")
    assert sub is not None, "sandbox reported success but produced no submission"

    ok, msg = validate(sub, "val", load_task("kuairand-pure"))
    assert ok, f"submission rejected: {msg}"
