"""Search policy: what to try next. Owner: A.

Pure functions over the tree — no I/O, no clock, no LLM. That is what makes the
policy unit-testable, and the policy is where the run either explores usefully
or spends 50 iterations polishing one dead end.

Priority order, highest first:

1. **debug-first.** Any leaf that failed and still has repair attempts left.
   A broken program is the cheapest thing in the tree to make valuable.
2. **draft phase.** Until `n_drafts` independent drafts exist, keep drafting
   from a different angle each time. Independent drafts are the only cheap
   protection against a bad first program anchoring the whole search.
3. **rescue.** No scored node and nothing repairable: draft again.
4. **explore.** After `explore_after` scored iterations with no real improvement,
   improve the *second-best distinct* node instead of the best one; if there is
   no distinct alternative, draft a fresh angle.

   This rule is only reachable while **`explore_after < conv_n`**, and the caller
   is responsible for that (`taskspec` rejects a config where it does not hold).
   `best_history` is monotone non-decreasing, so `converged()` can only fire once
   each of the last `conv_n` iterations was flat — by which point
   `flat_iters >= conv_n`. Set the two equal and the orchestrator stops on exactly
   the iteration this branch first becomes reachable, so it never runs: a plateau
   ends the run instead of redirecting the search, which is the opposite of the
   intent. Both defaulted to 3 for the whole of the first live campaign.
5. **greedy improve** on the best node.

Ties inside seed noise (0.0008) go to the *simpler* node. Validation is not the
score we are ranked on; the simpler of two statistically indistinguishable
programs generalises better to the hidden test set.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from orchestrator.contracts import Node, NodeKind

#: Five-seed std of the official baseline. Differences below this are noise.
SEED_NOISE = 0.0008

DRAFT_ANGLES: tuple[str, ...] = (
    (
        "Reproduce the official baseline first: a factorization machine over the "
        "categorical fields, trained on train only. Correctness before cleverness."
    ),
    (
        "Skip embeddings entirely. Build strong count and rate features (smoothed "
        "user CTR, item CTR, impression counts, recency) and fit a linear or "
        "gradient-boosted scorer over them."
    ),
    (
        "Attack the metric directly: GAUC and nDCG@5 are per-user ranking measures, "
        "so build a model whose scores are calibrated within each user rather than "
        "globally."
    ),
)


@dataclass(frozen=True)
class Action:
    kind: NodeKind
    parent_id: str | None
    reason: str  # journalled, so a judge can read why the agent moved where it did
    draft_angle: str | None = None


def _depth(nodes: dict[str, Node], node: Node) -> int:
    d, cur = 0, node
    while cur.parent_id is not None and cur.parent_id in nodes:
        cur = nodes[cur.parent_id]
        d += 1
    return d


def _code_len(node: Node) -> int:
    return len(node.proposal.code) if node.proposal else 0


def ok_nodes(nodes: dict[str, Node]) -> list[Node]:
    """Only nodes that ran, validated and scored may influence anything."""
    return [n for n in nodes.values() if n.status == "ok" and n.metrics]


def rank_nodes(nodes: dict[str, Node], *, noise: float = SEED_NOISE) -> list[Node]:
    """Scored nodes, best first, ties inside `noise` broken toward simplicity.

    Greedy clustering against the running leader rather than fixed buckets: two
    nodes 0.0004 apart must always be noise-equivalent, whichever side of a
    bucket boundary they happen to fall on.
    """
    remaining = sorted(ok_nodes(nodes), key=lambda n: -(n.primary or 0.0))
    ordered: list[Node] = []
    while remaining:
        leader = remaining[0]
        tied = [n for n in remaining if (leader.primary or 0.0) - (n.primary or 0.0) <= noise]
        tied.sort(key=lambda n: (_code_len(n), _depth(nodes, n), n.id))
        ordered.extend(tied)
        tied_ids = {n.id for n in tied}
        remaining = [n for n in remaining if n.id not in tied_ids]
    return ordered


def best_node(nodes: dict[str, Node], *, noise: float = SEED_NOISE) -> Node | None:
    ranked = rank_nodes(nodes, noise=noise)
    return ranked[0] if ranked else None


def repairable(nodes: dict[str, Node], *, max_repairs: int = 3) -> list[Node]:
    """Failed *leaves* with attempts left. Following leaves walks the repair
    chain n003 -> n004 -> n005 without ever re-repairing an abandoned node."""
    return [
        n
        for n in nodes.values()
        if n.status in ("buggy", "timeout")
        and n.repair_attempts < max_repairs
        and not n.children
    ]


def drafts(nodes: dict[str, Node]) -> list[Node]:
    return [n for n in nodes.values() if n.kind == "draft"]


def next_action(
    nodes: dict[str, Node],
    *,
    flat_iters: int = 0,
    n_drafts: int = 3,
    max_repairs: int = 3,
    explore_after: int = 2,
    noise: float = SEED_NOISE,
    angles: Sequence[str] = DRAFT_ANGLES,
) -> Action:
    """Pick the next (kind, parent). See the module docstring for the order."""
    # 1. debug-first
    broken = repairable(nodes, max_repairs=max_repairs)
    if broken:
        target = max(broken, key=lambda n: (n.iteration, n.id))
        return Action(
            "debug",
            target.id,
            f"{target.id} failed with error_class="
            f"{target.exec_result.error_class if target.exec_result else 'unknown'}; "
            f"repair attempt {target.repair_attempts + 1}/{max_repairs}",
        )

    n_drafted = len(drafts(nodes))
    scored = ok_nodes(nodes)

    # 2. draft phase
    if n_drafted < n_drafts:
        return Action(
            "draft",
            None,
            f"draft phase {n_drafted + 1}/{n_drafts}: independent angles guard against a "
            f"weak first program anchoring the whole search",
            draft_angle=angles[n_drafted % len(angles)],
        )

    # 3. rescue — everything drafted so far is dead
    if not scored:
        return Action(
            "draft",
            None,
            "no node has ever scored and nothing is repairable; drafting a fresh angle",
            draft_angle=angles[n_drafted % len(angles)],
        )

    ranked = rank_nodes(nodes, noise=noise)
    best = ranked[0]

    # 4. explore after a flat stretch
    if flat_iters >= explore_after:
        alternative = next(
            (n for n in ranked[1:] if (best.primary or 0) - (n.primary or 0) > noise),
            None,
        )
        if alternative is not None:
            return Action(
                "improve",
                alternative.id,
                f"{flat_iters} scored iterations without real improvement on {best.id}; "
                f"exploring the second-best distinct node {alternative.id} "
                f"(primary {alternative.primary}) instead",
            )
        return Action(
            "draft",
            None,
            f"{flat_iters} flat iterations and no distinct alternative in the tree; "
            f"drafting a fresh angle rather than polishing {best.id}",
            draft_angle=angles[n_drafted % len(angles)],
        )

    # 5. greedy improve
    return Action(
        "improve",
        best.id,
        f"greedy improve on the validation-best node {best.id} (primary {best.primary})",
    )


def is_flat(before: float | None, after: float | None, eps: float) -> bool:
    """Did this scored iteration fail to move the best by more than eps?"""
    if after is None:
        return True
    if before is None:
        return False
    return (after - before) <= eps


_FP_TOL = 1e-9  # metrics are 4dp; never let float error decide when a 6h run stops


def converged(best_history: Iterable[float], *, eps: float, n: int) -> bool:
    """No improvement > eps over the last `n` **scored** iterations.

    `best_history` is best-so-far validation primary after each scored
    iteration. Errored iterations never append to it, so three crashes in a row
    can never be mistaken for a plateau.
    """
    hist = list(best_history)
    if len(hist) <= n:
        return False
    # The rule is "not improved by MORE than eps", so an improvement of exactly eps
    # converges. Binary floating point makes 0.6020 - 0.6000 == 0.002000000000000002,
    # which fails a bare <= and silently keeps a converged run iterating - burning
    # wall-clock and tokens, both of which are scored. Compare with a tolerance.
    return (hist[-1] - hist[-1 - n]) - eps <= _FP_TOL
