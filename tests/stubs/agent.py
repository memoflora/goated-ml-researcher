"""StubAgent — canned proposals, no network. Mirrors B's `agent.py` seam.

    draft(ctx) -> Proposal
    improve(ctx, parent) -> Proposal
    repair(ctx, node) -> Proposal

Token counts are deterministic and roughly proportional to what really goes on
the wire, so accounting assertions in tests mean something.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import Context, Node, Proposal

_TMPL = Path(__file__).resolve().parents[1] / "fixtures" / "stub_pipeline.py.tmpl"

DRAFT_ANGLES = (
    "reproduce the official FM baseline as faithfully as possible",
    "start from a simple popularity + user-CTR prior, no embeddings",
    "start from a gradient-boosted tree over hand-built count features",
)


def render_pipeline(variant: str, n_rows: int = 1000) -> str:
    return (
        _TMPL.read_text(encoding="utf-8")
        .replace("__VARIANT__", variant)
        .replace("__N_ROWS__", str(n_rows))
    )


class StubAgent:
    """Deterministic stand-in for the real LLM client."""

    model = "stub-agent-v1"

    def __init__(self, *, n_rows: int = 1000, empty_hypothesis_once: bool = False) -> None:
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
        code = render_pipeline(variant, self.n_rows)
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
