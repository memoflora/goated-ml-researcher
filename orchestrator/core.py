"""The loop: tree, convergence, budget, checkpoint/resume, finalisation. Owner: A.

If a run stalls, crashes, or needs a human, it is a bug in this file.

The loop is exactly contracts.md §4:

    for iteration in 1..50, while wall_clock < 6h and not converged:
        kind, parent = policy.next_action(tree)
        proposal     = agent.draft|improve|repair(ctx, parent)
        write proposal.code -> node.workspace/pipeline.py
        result       = sandbox.run(node, split="val", ...)
        on failure   -> journal, repair up to 3x, then dead, then route around
        on success   -> validate, score, update best, check convergence

Every component is injected and duck-typed against the frozen seams, so this
file runs identically against the stubs and against the real modules.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from orchestrator import journal as journal_mod
from orchestrator import policy
from orchestrator.contracts import (
    PIPELINE_CLI,
    Budget,
    Context,
    ExecResult,
    HistoryEntry,
    Idea,
    Node,
    NodeKind,
    Proposal,
    TaskSpec,
    primary,
)
from orchestrator.journal import Accountant

#: Fallback whitelist when C's requirements-pipeline.txt is not on disk yet.
DEFAULT_WHITELIST = ["numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "torch"]

#: Run modes, contracts.md §7.
MODES: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_iters": 3,
        "subsample": 0.02,
        "stub_llm": True,
        "timeout_s": 120,
        "wall_clock_s": 300,
        "token_budget": None,
        "iter_estimate_s": 5.0,
        "final_seeds": (0,),
    },
    "dev": {
        "max_iters": 8,
        "subsample": 0.2,
        "stub_llm": False,
        "timeout_s": 600,
        "wall_clock_s": 3600,
        "token_budget": 400_000,
        "iter_estimate_s": 240.0,
        "final_seeds": (0,),
    },
    "official": {
        "max_iters": 50,
        "subsample": None,
        "stub_llm": False,
        "timeout_s": 1500,
        "wall_clock_s": 6 * 3600,
        "token_budget": 4_000_000,
        "iter_estimate_s": 420.0,
        "final_seeds": (0, 1, 2),
    },
}

MAX_HISTORY = 10  # compact history entries carried into a prompt
MAX_HYPOTHESIS_CHARS = 260  # per history entry — never the whole paragraph


# ---------------------------------------------------------------------------
# seams (documentation only — everything is duck-typed)
# ---------------------------------------------------------------------------


class AgentLike(Protocol):
    def draft(self, ctx: Context) -> Proposal: ...
    def improve(self, ctx: Context, parent: Node) -> Proposal: ...
    def repair(self, ctx: Context, node: Node) -> Proposal: ...


class ExecutorLike(Protocol):
    def run(
        self,
        node: Node,
        *,
        split: str,
        seed: int,
        timeout_s: int,
        subsample: float | None = None,
    ) -> ExecResult: ...


class EvaluatorLike(Protocol):
    def score(self, submission: Path, split: str) -> dict[str, float]: ...
    def validate(self, submission: Path, split: str) -> tuple[bool, str]: ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def new_run_id(now: datetime | None = None, runs_dir: Path | str | None = None) -> str:
    """Minute-resolution id, per contracts.md §6, made collision-proof.

    Two runs launched inside the same minute must never share a directory: the
    second would append to the first's journal, and the journal is graded.
    """
    stem = (now or datetime.now(timezone.utc)).strftime("r%Y%m%d-%H%M")
    if runs_dir is None:
        return stem
    root = Path(runs_dir)
    if not (root / stem).exists():
        return stem
    n = 2
    while (root / f"{stem}-{n}").exists():
        n += 1
    return f"{stem}-{n}"


def atomic_write(path: Path, text: str) -> None:
    """temp -> fsync -> rename. A kill -9 leaves either the old file or the new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - a missing git is not a run failure
        return "unknown"


def _delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {k: round(v - baseline[k], 4) for k, v in metrics.items() if k in baseline}


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------


@dataclass
class Tree:
    nodes: dict[str, Node] = field(default_factory=dict)
    counter: int = 0

    def add(
        self,
        *,
        parent_id: str | None,
        kind: NodeKind,
        iteration: int,
        workspace_root: Path,
        proposal: Proposal | None = None,
        repair_attempts: int = 0,
    ) -> Node:
        node_id = f"n{self.counter:03d}"
        self.counter += 1
        node = Node(
            id=node_id,
            parent_id=parent_id,
            kind=kind,
            iteration=iteration,
            workspace=Path(workspace_root) / "nodes" / node_id,
            proposal=proposal,
            repair_attempts=repair_attempts,
        )
        self.nodes[node_id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(node_id)
        return node

    def get(self, node_id: str | None) -> Node | None:
        return self.nodes.get(node_id) if node_id else None

    def best(self) -> Node | None:
        return policy.best_node(self.nodes)

    def ancestors(self, node: Node) -> list[Node]:
        chain, cur = [], node
        while cur.parent_id and cur.parent_id in self.nodes:
            cur = self.nodes[cur.parent_id]
            chain.append(cur)
        return chain

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"counter": self.counter, "nodes": {k: _node_to_dict(v) for k, v in self.nodes.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> Tree:
        tree = cls(counter=int(data.get("counter", 0)))
        tree.nodes = {k: _node_from_dict(v) for k, v in data.get("nodes", {}).items()}
        return tree


def _node_to_dict(n: Node) -> dict:
    return {
        "id": n.id,
        "parent_id": n.parent_id,
        "kind": n.kind,
        "iteration": n.iteration,
        "workspace": str(n.workspace),
        "status": n.status,
        "repair_attempts": n.repair_attempts,
        "children": list(n.children),
        "metrics": n.metrics,
        "proposal": (
            {
                "hypothesis": n.proposal.hypothesis,
                "plan": n.proposal.plan,
                "code": n.proposal.code,
                "idea_ids": n.proposal.idea_ids,
                "tokens_in": n.proposal.tokens_in,
                "tokens_out": n.proposal.tokens_out,
                "model": n.proposal.model,
            }
            if n.proposal
            else None
        ),
        "exec_result": (
            {
                "ok": n.exec_result.ok,
                "exit_code": n.exec_result.exit_code,
                "stdout_tail": n.exec_result.stdout_tail,
                "stderr_tail": n.exec_result.stderr_tail,
                "error_class": n.exec_result.error_class,
                "error_excerpt": n.exec_result.error_excerpt,
                "result_json": n.exec_result.result_json,
                "artifacts": {k: str(v) for k, v in n.exec_result.artifacts.items()},
                "wall_s": n.exec_result.wall_s,
                "peak_rss_mb": n.exec_result.peak_rss_mb,
            }
            if n.exec_result
            else None
        ),
    }


def _node_from_dict(d: dict) -> Node:
    node = Node(
        id=d["id"],
        parent_id=d.get("parent_id"),
        kind=d["kind"],
        iteration=int(d.get("iteration", 0)),
        workspace=Path(d["workspace"]),
        status=d.get("status", "pending"),
        repair_attempts=int(d.get("repair_attempts", 0)),
        children=list(d.get("children", [])),
        metrics=d.get("metrics"),
    )
    if d.get("proposal"):
        node.proposal = Proposal(**d["proposal"])
    if d.get("exec_result"):
        raw = dict(d["exec_result"])
        raw["artifacts"] = {k: Path(v) for k, v in (raw.get("artifacts") or {}).items()}
        node.exec_result = ExecResult(**raw)
    return node


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Drives one run to convergence, the iteration cap, or the wall clock.

    Never prompts. There is no interactive mode: wanting one is a signal to
    make the agent handle the case instead.
    """

    def __init__(
        self,
        task: TaskSpec,
        *,
        run_dir: Path,
        run_id: str,
        agent: AgentLike,
        executor: ExecutorLike,
        evaluator: EvaluatorLike,
        knowledge: Any | None = None,
        datacard: Any | None = None,
        mode: str = "smoke",
        seed: int = 0,
        subsample: float | None = None,
        timeout_s: int | None = None,
        token_budget: int | None = None,
        n_drafts: int = 3,
        max_repairs: int = 3,
        final_seeds: tuple[int, ...] | None = None,
        iter_estimate_s: float | None = None,
        components: dict[str, str] | None = None,
        journal: journal_mod.Journal | None = None,
    ) -> None:
        cfg = MODES.get(mode, MODES["smoke"])
        self.task = task
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.agent = agent
        self.executor = executor
        self.evaluator = evaluator
        self.knowledge = knowledge
        self.datacard = datacard
        self.mode = mode
        self.seed = seed
        self.subsample = cfg["subsample"] if subsample is None else subsample
        self.timeout_s = int(timeout_s if timeout_s is not None else cfg["timeout_s"])
        self.n_drafts = n_drafts
        self.max_repairs = max_repairs
        self.final_seeds = tuple(final_seeds if final_seeds is not None else cfg["final_seeds"])
        self._iter_estimate_default = float(
            iter_estimate_s if iter_estimate_s is not None else cfg["iter_estimate_s"]
        )

        self.components = components or {}
        self.tree = Tree()
        self.iteration = 0
        self.best_id: str | None = None
        self.best_history: list[float] = []  # best-so-far after each SCORED iteration
        self.flat_iters = 0
        self.stop_reason: str | None = None
        self.iter_durations: list[float] = []
        self.acct = Accountant(
            token_budget=token_budget if token_budget is not None else cfg["token_budget"]
        )
        self.journal = journal or journal_mod.configure(self.run_dir, run_id)
        self._data_card: str | None = None
        self._whitelist: list[str] | None = None

    # -- public ----------------------------------------------------------

    def run(self) -> dict:
        """Run to a stopping condition and return the run summary."""
        self._prepare_run_dir()
        self._emit_run_start()
        try:
            while True:
                stop = self._should_stop()
                if stop:
                    self.stop_reason = self.stop_reason or stop
                    break
                self.iteration += 1
                t0 = time.monotonic()
                self._run_iteration()
                self.iter_durations.append(time.monotonic() - t0)
                self.acct.tick_iteration()
                self.checkpoint()
        except KeyboardInterrupt:  # pragma: no cover - operator abort
            self.stop_reason = "interrupted"
            self.checkpoint()
            raise
        return self._finish()

    @classmethod
    def resume(cls, run_dir: Path, run_id: str, **kwargs: Any) -> Orchestrator:
        """Rebuild a run from its checkpoint. No node is lost, no human needed."""
        run_dir = Path(run_dir)
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        task = kwargs.pop("task", None) or task_from_config(cfg)
        kwargs.setdefault("mode", cfg.get("mode", "smoke"))
        kwargs.setdefault("seed", cfg.get("seed", 0))
        orch = cls(task, run_dir=run_dir, run_id=run_id, **kwargs)
        orch.tree = Tree.from_dict(state["tree"])
        orch.iteration = int(state.get("iteration", 0))
        orch.best_id = state.get("best_id")
        orch.best_history = list(state.get("best_history", []))
        orch.flat_iters = int(state.get("flat_iters", 0))
        orch.iter_durations = list(state.get("iter_durations", []))
        orch.acct.restore(state.get("accounting", {}), elapsed_s=state.get("elapsed_s", 0.0))
        orch.journal.emit(
            {
                "event": "recovery",
                "iteration": orch.iteration,
                "recovery": "resumed",
                "hypothesis": (
                    f"Resumed {run_id} from checkpoint at iteration {orch.iteration} with "
                    f"{len(orch.tree.nodes)} nodes; no state was lost and no human was needed."
                ),
            }
        )
        return orch

    # -- one iteration ---------------------------------------------------

    def _run_iteration(self) -> None:
        action = policy.next_action(
            self.tree.nodes,
            flat_iters=self.flat_iters,
            n_drafts=self.n_drafts,
            max_repairs=self.max_repairs,
        )
        target = self.tree.get(action.parent_id)

        # A debug node inherits the repair counter, so the chain n003 -> n004 ->
        # n005 dies after exactly max_repairs attempts however deep it goes.
        repair_attempts = (target.repair_attempts + 1) if (action.kind == "debug" and target) else 0
        ctx = self.build_context(action, target, repair_attempts=repair_attempts)

        try:
            if action.kind == "draft":
                proposal = self.agent.draft(ctx)
            elif action.kind == "improve":
                assert target is not None
                proposal = self.agent.improve(ctx, target)
            else:
                assert target is not None
                proposal = self.agent.repair(ctx, target)
        except Exception as exc:  # noqa: BLE001 - the LLM call failed after B's retries
            self.journal.emit(
                {
                    "event": "error",
                    "iteration": self.iteration,
                    "node_id": None,
                    "error_class": "unknown",
                    "recovery": "skip_iteration",
                    "hypothesis": f"agent call failed: {type(exc).__name__}: {exc}",
                }
            )
            return

        self.acct.add_tokens(proposal.tokens_in, proposal.tokens_out)
        node = self.tree.add(
            parent_id=action.parent_id,
            kind=action.kind,
            iteration=self.iteration,
            workspace_root=self.run_dir,
            proposal=proposal,
            repair_attempts=repair_attempts,
        )
        node.status = "running"
        node.workspace.mkdir(parents=True, exist_ok=True)
        (node.workspace / "pipeline.py").write_text(proposal.code, encoding="utf-8")

        self.journal.emit(
            {
                "event": "proposal",
                "iteration": self.iteration,
                "node_id": node.id,
                "parent_id": node.parent_id,
                "kind": node.kind,
                "hypothesis": proposal.hypothesis,
                "plan": proposal.plan,
                "idea_ids": proposal.idea_ids,
                "tokens_in": proposal.tokens_in,
                "tokens_out": proposal.tokens_out,
                "model": proposal.model,
                "reason": action.reason,
                "context_chars": ctx_size(ctx),
            }
        )

        result = self.executor.run(
            node,
            split="val",
            seed=self.seed,
            timeout_s=self.timeout_s,
            subsample=self.subsample,
        )
        node.exec_result = result
        self.acct.add_exec(result.wall_s)
        self.journal.emit(
            {
                "event": "exec",
                "iteration": self.iteration,
                "node_id": node.id,
                "kind": node.kind,
                "error_class": result.error_class,
                "wall_s": round(result.wall_s, 2),
                "exit_code": result.exit_code,
                "peak_rss_mb": round(result.peak_rss_mb, 1),
                "result_json": result.result_json,
            }
        )

        if not result.ok:
            self._handle_failure(node, result.error_class or "unknown", result.error_excerpt or "")
            return

        submission = result.artifacts.get("submission")
        if submission is None:
            self._handle_failure(node, "contract", "pipeline exited 0 but produced no submission")
            return

        ok, msg = self.evaluator.validate(submission, "val")
        if not ok:
            self._handle_failure(node, "contract", f"submission failed validation: {msg}")
            return

        try:
            metrics = dict(self.evaluator.score(submission, "val"))
        except Exception as exc:  # noqa: BLE001 - a scorer crash is the node's problem
            self._handle_failure(node, "eval", f"scoring raised {type(exc).__name__}: {exc}")
            return
        metrics.setdefault("primary", primary(metrics))
        node.metrics = metrics
        node.status = "ok"

        self.journal.emit(
            {
                "event": "eval",
                "iteration": self.iteration,
                "node_id": node.id,
                "parent_id": node.parent_id,
                "kind": node.kind,
                "metrics": metrics,
                "delta_vs_baseline": _delta(metrics, self.task.baseline_val),
            }
        )
        self._record_scored(node)

    def _handle_failure(self, node: Node, error_class: str, excerpt: str) -> None:
        """Three repairs, then dead, then route around. This is the Robustness score."""
        node.status = "timeout" if error_class == "timeout" else "buggy"
        self.journal.emit(
            {
                "event": "error",
                "iteration": self.iteration,
                "node_id": node.id,
                "parent_id": node.parent_id,
                "kind": node.kind,
                "error_class": error_class,
                "error_excerpt": excerpt[:1500],
            }
        )
        if node.repair_attempts < self.max_repairs:
            self.journal.emit(
                {
                    "event": "recovery",
                    "iteration": self.iteration,
                    "node_id": node.id,
                    "error_class": error_class,
                    "recovery": "schedule_debug",
                    "repair_attempt": node.repair_attempts + 1,
                    "max_repairs": self.max_repairs,
                }
            )
            return
        node.status = "dead"
        self.journal.emit(
            {
                "event": "prune",
                "iteration": self.iteration,
                "node_id": node.id,
                "error_class": error_class,
                "recovery": "route_around",
                "hypothesis": (
                    f"{node.id} failed {self.max_repairs} repair attempts on error_class="
                    f"{error_class}; marking it dead and continuing from the best live node."
                ),
            }
        )

    def _record_scored(self, node: Node) -> None:
        """Update best, the flat streak and the convergence window."""
        prev_best = self.best_history[-1] if self.best_history else None
        best = self.tree.best()
        best_primary = best.primary if best else None
        assert best_primary is not None

        if best is not None and best.id != self.best_id:
            self.best_id = best.id
            self.journal.emit(
                {
                    "event": "best_updated",
                    "iteration": self.iteration,
                    "node_id": best.id,
                    "metrics": best.metrics,
                    "delta_vs_baseline": _delta(best.metrics or {}, self.task.baseline_val),
                }
            )

        if policy.is_flat(prev_best, best_primary, self.task.conv_eps):
            self.flat_iters += 1
        else:
            self.flat_iters = 0
        self.best_history.append(best_primary)

    # -- context ---------------------------------------------------------

    def build_context(
        self, action: policy.Action, target: Node | None, *, repair_attempts: int = 0
    ) -> Context:
        """Assemble the prompt inputs. This is where token cost is won or lost.

        Carries: task card, C's data card, the parent's full code, the parent's
        metrics, D's top-K ideas, and a compact history — hypothesis plus metric
        delta only, **never** past code.
        """
        parent_code = parent_metrics = parent_hypothesis = None
        error_class = error_excerpt = stderr_tail = None
        prior_plans: list[str] = []

        if target is not None:
            src = target
            if action.kind == "debug":
                # Repair sees the failing program and every plan already tried on
                # this chain, so attempt 3 is never a re-run of attempt 1.
                res = target.exec_result
                error_class = res.error_class if res else None
                error_excerpt = res.error_excerpt if res else None
                stderr_tail = (res.stderr_tail[-1500:] if res and res.stderr_tail else None)
                for anc in [target, *self.tree.ancestors(target)]:
                    if anc.kind == "debug" and anc.proposal:
                        prior_plans.extend(anc.proposal.plan)
            if src.proposal:
                parent_code = src.proposal.code
                parent_hypothesis = src.proposal.hypothesis
            parent_metrics = src.metrics

        return Context(
            task=self.task,
            run_id=self.run_id,
            iteration=self.iteration,
            data_card=self.data_card(),
            ideas=self.ideas(),
            history=self.compact_history(),
            budget=self.budget(),
            library_whitelist=self.whitelist(),
            pipeline_cli=PIPELINE_CLI,
            baseline_val=dict(self.task.baseline_val),
            parent_code=parent_code,
            parent_metrics=parent_metrics,
            parent_hypothesis=parent_hypothesis,
            error_class=error_class,
            error_excerpt=error_excerpt,
            stderr_tail=stderr_tail,
            prior_repair_plans=prior_plans,
            repair_attempt=repair_attempts,
            draft_angle=action.draft_angle,
        )

    def compact_history(self, limit: int = MAX_HISTORY) -> list[HistoryEntry]:
        """Hypothesis + metric delta only. Past code never travels."""
        nodes = sorted(self.tree.nodes.values(), key=lambda n: (n.iteration, n.id))
        out: list[HistoryEntry] = []
        for n in nodes[-limit:]:
            hyp = n.proposal.hypothesis if n.proposal else ""
            p = n.primary
            out.append(
                HistoryEntry(
                    iteration=n.iteration,
                    node_id=n.id,
                    kind=n.kind,
                    hypothesis=hyp[:MAX_HYPOTHESIS_CHARS],
                    primary=p,
                    delta_vs_baseline=(
                        round(p - self.task.baseline_val["primary"], 4) if p is not None else None
                    ),
                    status=n.status,
                    error_class=(n.exec_result.error_class if n.exec_result else None),
                )
            )
        return out

    def budget(self) -> Budget:
        return Budget(
            iters_left=max(0, self.task.max_iters - self.iteration),
            seconds_left=max(0.0, self.task.wall_clock_s - self.acct.elapsed_s()),
            tokens_left=self.acct.tokens_left(),
            tokens_in_used=self.acct.tokens_in,
            tokens_out_used=self.acct.tokens_out,
        )

    def data_card(self) -> str:
        if self._data_card is None:
            try:
                self._data_card = self.datacard.data_card() if self.datacard else ""
            except Exception as exc:  # noqa: BLE001 - run without a data card rather than not at all
                self._data_card = ""
                self.journal.emit(
                    {
                        "event": "error",
                        "iteration": self.iteration,
                        "error_class": "data",
                        "error_excerpt": f"data_card() failed: {type(exc).__name__}: {exc}",
                        "recovery": "empty_data_card",
                    }
                )
            self.journal.emit(
                {
                    "event": "data_card",
                    "iteration": self.iteration,
                    "chars": len(self._data_card),
                    "approx_tokens": len(self._data_card) // 4,
                }
            )
        return self._data_card

    def ideas(self, k: int = 5) -> list[Idea]:
        if self.knowledge is None:
            return []
        best = self.tree.best()
        try:
            return list(
                self.knowledge.retrieve(
                    tried=sorted(
                        {
                            i
                            for n in self.tree.nodes.values()
                            if n.proposal
                            for i in n.proposal.idea_ids
                        }
                    ),
                    best_metrics=dict(best.metrics or {}) if best else {},
                    budget_left=max(0, self.task.max_iters - self.iteration),
                    k=k,
                )
            )
        except Exception as exc:  # noqa: BLE001 - no ideas is survivable, a crash is not
            self.journal.emit(
                {
                    "event": "error",
                    "iteration": self.iteration,
                    "error_class": "unknown",
                    "error_excerpt": f"retrieve() failed: {type(exc).__name__}: {exc}",
                    "recovery": "no_ideas_this_iteration",
                }
            )
            return []

    def whitelist(self) -> list[str]:
        if self._whitelist is None:
            req = Path("requirements-pipeline.txt")
            if req.exists():
                self._whitelist = [
                    ln.strip()
                    for ln in req.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")
                ]
            else:
                self._whitelist = list(DEFAULT_WHITELIST)
        return self._whitelist

    # -- stopping --------------------------------------------------------

    def iteration_estimate_s(self) -> float:
        if len(self.iter_durations) >= 2:
            return max(statistics.median(self.iter_durations), 1.0)
        return self._iter_estimate_default

    def _should_stop(self) -> str | None:
        """Refuse to start an iteration that cannot finish inside the ceiling."""
        if policy.converged(self.best_history, eps=self.task.conv_eps, n=self.task.conv_n):
            return "converged"
        if self.iteration >= self.task.max_iters:
            return "max_iters"
        est = self.iteration_estimate_s()
        reserve = est * max(1, len(self.final_seeds)) + 30.0
        if self.acct.elapsed_s() + est + reserve > self.task.wall_clock_s:
            return "wall_clock"
        if self.acct.tokens_left() <= 0:
            return "token_budget"
        return None

    # -- checkpoint ------------------------------------------------------

    def checkpoint(self) -> None:
        atomic_write(
            self.run_dir / "state.json",
            json.dumps(
                {
                    "run_id": self.run_id,
                    "mode": self.mode,
                    "iteration": self.iteration,
                    "best_id": self.best_id,
                    "best_history": self.best_history,
                    "flat_iters": self.flat_iters,
                    "iter_durations": self.iter_durations,
                    "stop_reason": self.stop_reason,
                    "elapsed_s": round(self.acct.elapsed_s(), 3),
                    "accounting": self.acct.snapshot(),
                    "tree": self.tree.to_dict(),
                },
                indent=1,
            ),
        )

    def _prepare_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "nodes").mkdir(exist_ok=True)
        interventions = self.run_dir / "interventions.md"
        if not interventions.exists():
            atomic_write(
                interventions,
                f"# Manual interventions — {self.run_id}\n\n"
                "Every human touch during this run, timestamped. The count is directly\n"
                "scored under Autonomy. An empty table is the goal.\n\n"
                "| UTC | Who | What they did | Why the agent could not |\n"
                "|---|---|---|---|\n",
            )
        atomic_write(
            self.run_dir / "config.json",
            json.dumps(
                {
                    "run_id": self.run_id,
                    "mode": self.mode,
                    "seed": self.seed,
                    "git_sha": git_sha(),
                    "started_utc": self.acct.wall_started_utc,
                    "components": self.components,
                    "subsample": self.subsample,
                    "timeout_s": self.timeout_s,
                    "final_seeds": list(self.final_seeds),
                    "token_budget": self.acct.token_budget,
                    "task": {
                        "name": self.task.name,
                        "data_dir": str(self.task.data_dir),
                        "metrics": list(self.task.metrics),
                        "baseline_val": self.task.baseline_val,
                        "baseline_test": self.task.baseline_test,
                        "ceiling": self.task.ceiling,
                        "max_iters": self.task.max_iters,
                        "wall_clock_s": self.task.wall_clock_s,
                        "conv_eps": self.task.conv_eps,
                        "conv_n": self.task.conv_n,
                    },
                },
                indent=1,
            ),
        )

    def _emit_run_start(self) -> None:
        self.journal.emit(
            {
                "event": "run_start",
                "iteration": 0,
                "mode": self.mode,
                "git_sha": git_sha(),
                "task": self.task.name,
                "max_iters": self.task.max_iters,
                "wall_clock_s": self.task.wall_clock_s,
                "conv_eps": self.task.conv_eps,
                "conv_n": self.task.conv_n,
                "baseline_val": self.task.baseline_val,
                "ceiling": self.task.ceiling,
                "subsample": self.subsample,
                "components": self.components,
            }
        )

    # -- finalisation ----------------------------------------------------

    def _finish(self) -> dict:
        best = self.tree.best()
        if self.stop_reason == "converged":
            self.journal.emit(
                {
                    "event": "converged",
                    "iteration": self.iteration,
                    "node_id": best.id if best else None,
                    "metrics": best.metrics if best else None,
                    "hypothesis": (
                        f"Validation primary has not improved by more than {self.task.conv_eps} "
                        f"over the last {self.task.conv_n} scored iterations."
                    ),
                }
            )
        final = self.finalize(best)
        summary = {
            "run_id": self.run_id,
            "stop_reason": self.stop_reason,
            "iterations": self.iteration,
            "max_iters": self.task.max_iters,
            "best_node": best.id if best else None,
            "best_metrics": best.metrics if best else None,
            "delta_vs_baseline": _delta(best.metrics or {}, self.task.baseline_val) if best else {},
            "final_submission": str(final["submission"]) if final.get("submission") else None,
            "final_valid": final.get("valid"),
            "nodes": len(self.tree.nodes),
            "dead_nodes": sum(1 for n in self.tree.nodes.values() if n.status == "dead"),
            **self.acct.snapshot(),
        }
        self.journal.emit({"event": "run_end", "iteration": self.iteration, **summary})
        self.checkpoint()
        atomic_write(self.run_dir / "summary.json", json.dumps(summary, indent=1))
        return summary

    def finalize(self, best: Node | None) -> dict:
        """Rerun the validation-best node on `--split test`, seed-averaged.

        Never select or submit on a single seed when we can afford not to: seed
        noise is 0.0008 and eps is 0.002.
        """
        out: dict[str, Any] = {"submission": None, "valid": None, "seeds": []}
        if best is None or best.proposal is None:
            self.journal.emit(
                {
                    "event": "error",
                    "iteration": self.iteration,
                    "error_class": "contract",
                    "error_excerpt": "no scored node to finalise; run produced no submission",
                }
            )
            return out

        best_dir = self.run_dir / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(best.workspace, best_dir)

        final_dir = self.run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []
        for seed in self.final_seeds:
            ws = final_dir / f"seed{seed}"
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "pipeline.py").write_text(best.proposal.code, encoding="utf-8")
            seed_node = Node(
                id=f"final-seed{seed}",
                parent_id=best.id,
                kind=best.kind,
                iteration=self.iteration,
                workspace=ws,
                proposal=best.proposal,
            )
            try:
                res = self.executor.run(
                    seed_node, split="test", seed=seed, timeout_s=self.timeout_s, subsample=None
                )
            except Exception as exc:  # noqa: BLE001 - one bad seed must not lose the run
                self.journal.emit(
                    {
                        "event": "error",
                        "iteration": self.iteration,
                        "node_id": seed_node.id,
                        "error_class": "runtime",
                        "error_excerpt": f"final seed {seed} raised {type(exc).__name__}: {exc}",
                        "recovery": "skip_seed",
                    }
                )
                continue
            self.acct.add_exec(res.wall_s)
            sub = res.artifacts.get("submission")
            if not res.ok or sub is None:
                self.journal.emit(
                    {
                        "event": "error",
                        "iteration": self.iteration,
                        "node_id": seed_node.id,
                        "error_class": res.error_class or "contract",
                        "error_excerpt": (res.error_excerpt or "no submission on test split")[:1500],
                        "recovery": "skip_seed",
                    }
                )
                continue
            ok, msg = self.evaluator.validate(sub, "test")
            if not ok:
                self.journal.emit(
                    {
                        "event": "error",
                        "iteration": self.iteration,
                        "node_id": seed_node.id,
                        "error_class": "contract",
                        "error_excerpt": f"final seed {seed} failed validation: {msg}",
                        "recovery": "skip_seed",
                    }
                )
                continue
            produced.append(sub)
            out["seeds"].append(seed)

        if not produced:
            self.journal.emit(
                {
                    "event": "error",
                    "iteration": self.iteration,
                    "error_class": "contract",
                    "error_excerpt": "every final-seed run failed; no test submission written",
                }
            )
            return out

        final_csv = final_dir / "submission.csv"
        rank_average(produced, final_csv)
        ok, msg = self.evaluator.validate(final_csv, "test")
        out["submission"], out["valid"] = final_csv, ok
        self.journal.emit(
            {
                "event": "eval",
                "iteration": self.iteration,
                "node_id": best.id,
                "split": "test",
                "recovery": None if ok else "final_submission_invalid",
                "seeds_averaged": out["seeds"],
                "hypothesis": (
                    f"Final submission: node {best.id}, retrained on train+validation and "
                    f"rank-averaged over seeds {out['seeds']}. Validator says: {msg}"
                ),
            }
        )
        return out


def rank_average(submissions: list[Path], out_path: Path) -> None:
    """Rank-average several seed submissions into one.

    Ranks, not raw scores: only relative order is scored, and ranks are immune
    to a seed whose scores happen to live on a different scale.
    """
    tables: list[list[float]] = []
    header = "row_id,user_id,video_id,score"
    keys: list[tuple[str, str]] = []
    for path in submissions:
        rows = path.read_text(encoding="utf-8").splitlines()
        header = rows[0]
        scores, ids = [], []
        for line in rows[1:]:
            if not line.strip():
                continue
            _row_id, user_id, video_id, score = line.split(",")
            scores.append(float(score))
            ids.append((user_id, video_id))
        tables.append(scores)
        if not keys:
            keys = ids
    n = min(len(t) for t in tables)
    ranked: list[list[float]] = []
    for scores in tables:
        order = sorted(range(n), key=lambda i: scores[i])
        rank = [0.0] * n
        for position, index in enumerate(order):
            rank[index] = position / max(1, n - 1)
        ranked.append(rank)
    lines = [header]
    for i in range(n):
        mean = sum(r[i] for r in ranked) / len(ranked)
        user_id, video_id = keys[i]
        lines.append(f"{i},{user_id},{video_id},{mean:.6f}")
    atomic_write(out_path, "\n".join(lines) + "\n")


def ctx_size(ctx: Context) -> int:
    """Characters of prompt-bound content. Logged so token cost stays visible."""
    return (
        len(ctx.data_card)
        + len(ctx.parent_code or "")
        + len(ctx.error_excerpt or "")
        + len(ctx.stderr_tail or "")
        + sum(len(i.summary) + len(i.title) for i in ctx.ideas)
        + sum(len(h.hypothesis) for h in ctx.history)
        + len(ctx.draft_angle or "")
    )


def task_from_config(cfg: dict) -> TaskSpec:
    """Rebuild the TaskSpec a run was started with, from its config.json."""
    t = cfg["task"]
    return TaskSpec(
        name=t["name"],
        data_dir=Path(t["data_dir"]),
        metrics=tuple(t["metrics"]),
        baseline_val=t["baseline_val"],
        baseline_test=t["baseline_test"],
        ceiling=t.get("ceiling", 0.8645),
        max_iters=t.get("max_iters", 50),
        wall_clock_s=t.get("wall_clock_s", 6 * 3600),
        conv_eps=t.get("conv_eps", 0.002),
        conv_n=t.get("conv_n", 3),
    )
