"""StubAgent — canned proposals, no network. Mirrors B's `agent.py` seam.

    draft(ctx) -> Proposal
    improve(ctx, parent) -> Proposal
    repair(ctx, node) -> Proposal

Token counts are deterministic and roughly proportional to what really goes on
the wire, so accounting assertions in tests mean something.
"""

from __future__ import annotations

import json
from pathlib import Path

from orchestrator.contracts import Context, Node, Proposal

_TMPL = Path(__file__).resolve().parents[1] / "fixtures" / "stub_pipeline.py.tmpl"

DRAFT_ANGLES = (
    "reproduce the official FM baseline as faithfully as possible",
    "start from a simple popularity + user-CTR prior, no embeddings",
    "start from a gradient-boosted tree over hand-built count features",
)

#: What the rendered pipeline is told about the task. Everything is optional — the
#: pipeline detects whatever is missing from the directory it is pointed at — but
#: passing it means the stub does not have to guess a schema it was handed for free.
DEFAULT_SPEC: dict = {
    "columns": ["row_id", "user_id", "video_id", "score"],
    "prediction_column": "score",
    "loader": "auto",
}


def spec_for(task) -> dict:
    """Derive the pipeline's spec from a `TaskSpec`.

    This is the whole fix for the misalignment bug. The stub used to be handed a row
    *count* and invent ids from it, which can align with nothing; it is now handed the
    submission schema and reads the ids out of the real split.
    """
    spec = dict(DEFAULT_SPEC)
    if task is None:
        return spec

    columns = getattr(task, "submission_columns", None)
    if columns:
        spec["columns"] = list(columns)
    spec["prediction_column"] = getattr(task, "prediction_column", None) or spec["columns"][-1]

    cfg = getattr(task, "config", None)
    if cfg is None:
        return spec

    data = getattr(cfg, "data", None)
    if data is None:
        return spec
    if getattr(data, "loader", "") == "starter_kit":
        spec["loader"] = "starter_kit"
    if getattr(data, "target", None):
        spec["target"] = data.target

    plan = getattr(data, "split", None)
    if plan is not None and getattr(plan, "strategy", None) == "date":
        ranges = getattr(plan, "ranges", None) or {}
        spec["ranges"] = {
            {"val": "valid"}.get(k, k): [int(lo), int(hi)] for k, (lo, hi) in ranges.items()
        }
        spec["date_column"] = getattr(plan, "date_column", None) or "date"
    if getattr(data, "files", None):
        spec["log_files"] = list(data.files.values())
    return spec


def render_pipeline(variant: str, n_rows: int = 1000, *, spec: dict | None = None) -> str:
    """Render the canned pipeline.

    `n_rows` is retained only so the old positional call keeps working; the pipeline no
    longer invents rows, it reads them. Passing a row count is exactly what made the old
    stub unable to align with any real split.
    """
    del n_rows
    payload = json.dumps(spec if spec is not None else DEFAULT_SPEC, sort_keys=True)
    return (
        _TMPL.read_text(encoding="utf-8")
        .replace("__VARIANT__", variant)
        .replace("__SPEC_JSON__", payload)
    )


class StubAgent:
    """Deterministic stand-in for the real LLM client."""

    model = "stub-agent-v1"

    def __init__(self, *, n_rows: int = 1000, empty_hypothesis_once: bool = False) -> None:
        #: Kept for callers that still pass it; the rendered pipeline reads its row count
        #: off the real split now, because a made-up one can align with nothing.
        self.n_rows = n_rows
        self.calls: list[str] = []
        self._empty_once = empty_hypothesis_once

    # -- seam ------------------------------------------------------------
    def draft(self, ctx: Context) -> Proposal:
        self.calls.append("draft")
        angle = ctx.draft_angle or DRAFT_ANGLES[len(self.calls) % len(DRAFT_ANGLES)]
        return self._proposal(
            ctx,
            hypothesis=(
                f"The scored metric rewards per-user ordering, so the first program should "
                f"establish a trustworthy floor rather than a clever one: {angle}. "
                f"Baseline validation primary is {ctx.baseline_val.get('primary')}, and we "
                f"cannot reason about any later change until we can reproduce it."
            ),
            plan=[f"draft: {angle}", "write submission.csv for the val split", "print RESULT_JSON"],
            variant=f"draft-{len(self.calls)}",
            idea_ids=[i.id for i in ctx.ideas[:1]],
        )

    def improve(self, ctx: Context, parent: Node) -> Proposal:
        self.calls.append("improve")
        idea = ctx.ideas[0] if ctx.ideas else None
        return self._proposal(
            ctx,
            hypothesis=(
                f"Parent {parent.id} reached primary {parent.primary}. "
                + (
                    f"{idea.title} targets the largest remaining gap: {idea.summary}"
                    if idea
                    else "The largest remaining gap is per-user calibration of the scores."
                )
            ),
            plan=["one focused change on top of the parent program"],
            variant=f"improve-of-{parent.id}",
            idea_ids=[idea.id] if idea else [],
        )

    def repair(self, ctx: Context, node: Node) -> Proposal:
        self.calls.append("repair")
        return self._proposal(
            ctx,
            hypothesis=(
                f"Node {node.id} failed with error_class={ctx.error_class}. The fix is "
                f"mechanical: restore the contract without changing the modelling idea, so "
                f"the hypothesis under test stays comparable to its siblings."
            ),
            plan=[f"fix {ctx.error_class}", "change nothing else"],
            variant=f"repair-{node.id}-{ctx.repair_attempt}",
            idea_ids=list(node.proposal.idea_ids) if node.proposal else [],
        )

    # -- internals -------------------------------------------------------
    def _proposal(
        self, ctx: Context, *, hypothesis: str, plan: list[str], variant: str, idea_ids: list[str]
    ) -> Proposal:
        if self._empty_once:
            self._empty_once = False
            hypothesis = ""
        code = render_pipeline(variant, spec=spec_for(getattr(ctx, "task", None)))
        prompt_chars = (
            len(ctx.data_card)
            + len(ctx.parent_code or "")
            + sum(len(i.summary) for i in ctx.ideas)
            + sum(len(h.hypothesis) for h in ctx.history)
            + 2000
        )
        return Proposal(
            hypothesis=hypothesis,
            plan=plan,
            code=code,
            idea_ids=idea_ids,
            tokens_in=prompt_chars // 4,
            tokens_out=len(code) // 4 + len(hypothesis) // 4,
            model=self.model,
        )
