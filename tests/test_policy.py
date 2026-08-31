"""Policy unit tests. Owner: A.

The policy is pure, so every acceptance criterion in roles.md that is about
*what the agent decides to do next* is testable here without a run directory,
an LLM or a clock.
"""

from pathlib import Path

import pytest

from orchestrator.contracts import ExecResult, Node, Proposal
from orchestrator.policy import (
    SEED_NOISE,
    best_node,
    converged,
    is_flat,
    next_action,
    rank_nodes,
    repairable,
)


def mk(
    node_id,
    *,
    kind="improve",
    parent=None,
    status="ok",
    primary=None,
    code="x" * 100,
    attempts=0,
    children=(),
    error_class=None,
    iteration=0,
):
    node = Node(
        id=node_id,
        parent_id=parent,
        kind=kind,
        iteration=iteration,
        workspace=Path("/tmp") / node_id,
        status=status,
        repair_attempts=attempts,
        children=list(children),
    )
    node.proposal = Proposal(
        hypothesis="h", plan=["p"], code=code, idea_ids=[], tokens_in=1, tokens_out=1, model="m"
    )
    if primary is not None:
        node.metrics = {"gauc": primary, "ndcg@5": primary, "primary": primary}
    if error_class:
        node.exec_result = ExecResult(
            ok=False,
            exit_code=1,
            stdout_tail="",
            stderr_tail="",
            error_class=error_class,
            error_excerpt="boom",
            result_json=None,
            artifacts={},
            wall_s=1.0,
            peak_rss_mb=1.0,
        )
    return node


def tree(*nodes):
    return {n.id: n for n in nodes}


# -- convergence ----------------------------------------------------------


def test_converged_plateau():
    # Four scored iterations that gain 0.0018 in total: below eps=0.002.
    assert converged([0.6000, 0.6010, 0.6015, 0.6018], eps=0.002, n=3)


def test_converged_noise_only():
    # best-so-far never decreases; a noisy plateau still converges.
    assert converged([0.6100, 0.6100, 0.6104, 0.6107], eps=0.002, n=3)


def test_not_converged_late_jump():
    # A flat stretch followed by a real jump must not be called convergence.
    assert not converged([0.6000, 0.6005, 0.6008, 0.6210], eps=0.002, n=3)


def test_not_converged_before_window_fills():
    assert not converged([0.60, 0.60, 0.60], eps=0.002, n=3)
    assert converged([0.60, 0.60, 0.60, 0.60], eps=0.002, n=3)


def test_errored_iterations_never_enter_the_window():
    # Three crashes in a row append nothing, so they cannot declare victory.
    history = [0.6000]
    assert not converged(history, eps=0.002, n=3)


def test_is_flat():
    assert is_flat(0.60, 0.6010, 0.002)
    assert not is_flat(0.60, 0.6030, 0.002)
    assert is_flat(0.60, None, 0.002)
    assert not is_flat(None, 0.60, 0.002)


# -- selection ------------------------------------------------------------


def test_best_ignores_unscored_and_broken_nodes():
    t = tree(
        mk("n000", primary=0.61),
        mk("n001", status="buggy", error_class="syntax"),
        mk("n002", status="dead"),
        mk("n003", status="ok", primary=None),  # ok but never scored
    )
    assert best_node(t).id == "n000"


def test_a_bad_node_cannot_poison_best():
    # A buggy node carrying stale metrics must still be ignored.
    bad = mk("n001", status="buggy", primary=0.99, error_class="eval")
    t = tree(mk("n000", primary=0.61), bad)
    assert best_node(t).id == "n000"


def test_within_noise_prefer_the_simpler_node():
    t = tree(
        mk("n000", primary=0.6200, code="x" * 8000),
        mk("n001", primary=0.6204, code="x" * 900),  # +0.0004 == inside noise
    )
    assert best_node(t).id == "n001"


def test_outside_noise_prefer_the_better_node():
    t = tree(
        mk("n000", primary=0.6200, code="x" * 900),
        mk("n001", primary=0.6260, code="x" * 8000),
    )
    assert best_node(t).id == "n001"
    assert [n.id for n in rank_nodes(t)] == ["n001", "n000"]


# -- action ordering ------------------------------------------------------


def test_first_action_is_a_draft():
    action = next_action({})
    assert action.kind == "draft" and action.parent_id is None and action.draft_angle


def test_draft_phase_runs_three_independent_angles():
    angles = set()
    t = {}
    for i in range(3):
        action = next_action(t)
        assert action.kind == "draft"
        angles.add(action.draft_angle)
        t[f"n{i:03d}"] = mk(f"n{i:03d}", kind="draft", primary=0.60 + i * 0.01)
    assert len(angles) == 3, "three drafts must come from three different angles"
    assert next_action(t).kind == "improve"


def test_debug_first_beats_drafting_and_improving():
    t = tree(
        mk("n000", kind="draft", primary=0.62),
        mk("n001", kind="draft", status="buggy", error_class="import", iteration=2),
    )
    action = next_action(t)
    assert action.kind == "debug" and action.parent_id == "n001"


def test_repair_chain_targets_the_leaf_not_the_original():
    t = tree(
        mk("n000", kind="draft", status="buggy", children=["n001"], error_class="syntax"),
        mk("n001", kind="debug", parent="n000", status="buggy", attempts=1,
           error_class="syntax", iteration=2),
    )
    assert repairable(t) == [t["n001"]]
    assert next_action(t).parent_id == "n001"


def test_a_node_out_of_repairs_is_not_picked_again():
    t = tree(mk("n000", kind="draft", status="buggy", attempts=3, error_class="oom"))
    assert repairable(t) == []
    assert next_action(t).kind == "draft"  # routes around it


def test_greedy_improve_targets_the_best_node():
    t = tree(
        mk("n000", kind="draft", primary=0.60),
        mk("n001", kind="draft", primary=0.63),
        mk("n002", kind="draft", primary=0.61),
    )
    action = next_action(t, flat_iters=0)
    assert action.kind == "improve" and action.parent_id == "n001"


def test_three_flat_iterations_trigger_explore_on_the_second_best():
    t = tree(
        mk("n000", kind="draft", primary=0.60),
        mk("n001", kind="draft", primary=0.63),
        mk("n002", kind="draft", primary=0.61),
    )
    action = next_action(t, flat_iters=3)
    assert action.kind == "improve" and action.parent_id == "n002"
    assert "explor" in action.reason


def test_explore_falls_back_to_a_fresh_draft_when_nothing_is_distinct():
    t = tree(
        mk("n000", kind="draft", primary=0.6300),
        mk("n001", kind="draft", primary=0.6304),
        mk("n002", kind="draft", primary=0.6301),
    )
    action = next_action(t, flat_iters=4)
    assert action.kind == "draft" and action.draft_angle


def test_rescue_drafts_when_every_node_is_dead():
    t = tree(*(mk(f"n{i:03d}", kind="draft", status="dead", attempts=3) for i in range(3)))
    assert next_action(t).kind == "draft"


@pytest.mark.parametrize("delta", [0.0, SEED_NOISE / 2, -SEED_NOISE / 2])
def test_noise_bucketing_is_symmetric(delta):
    t = tree(
        mk("n000", primary=0.6200, code="x" * 5000),
        mk("n001", primary=0.6200 + delta, code="x" * 500),
    )
    assert best_node(t).id == "n001"


# ----------------------------------------------------- explore reachability
#
# The bug these guard: `explore_after` and `conv_n` both defaulted to 3, so the
# orchestrator stopped on exactly the iteration the explore branch first became
# reachable. `test_three_flat_iterations_trigger_explore_on_the_second_best`
# passed throughout — it calls `next_action(flat_iters=3)` directly and proves
# the branch works. Nothing proved the state was reachable in a real run.


def replay(scores, *, eps, conv_n, explore_after):
    """Replay a scored trajectory exactly as `Orchestrator` bookkeeps it.

    Returns (first_explore_iteration, first_converged_iteration), 1-indexed and
    None when it never happens. Mirrors `core.py`: `_should_stop()` is evaluated
    at the *start* of an iteration, before `next_action` picks a move, so
    converging on iteration k means the explore branch on k never executes.
    """
    best_history: list[float] = []
    flat = 0
    prev_best = None
    first_explore = first_converged = None
    for i, s in enumerate(scores, start=1):
        # A run stops before it acts, so check convergence against what the
        # previous iteration left behind.
        if first_converged is None and converged(best_history, eps=eps, n=conv_n):
            first_converged = i
            break
        if first_explore is None and flat >= explore_after:
            first_explore = i
        best = s if prev_best is None else max(prev_best, s)
        flat = flat + 1 if is_flat(prev_best, best, eps) else 0
        best_history.append(best)
        prev_best = best
    else:
        if first_converged is None and converged(best_history, eps=eps, n=conv_n):
            first_converged = len(scores) + 1
    return first_explore, first_converged


def test_explore_is_reachable_before_convergence_on_the_real_plateau():
    """The gpt-5.1 run's own trajectory (docs/handover/01-results.md).

    Under the shipped defaults (3, 3) the run halts at the start of iteration 7,
    after 6 scored iterations — and flat_iters reaches the explore threshold at
    exactly that moment, so the branch never executes. The task now sets
    explore_after=2 < conv_n=4.
    """
    scores = [0.58458, 0.56031, 0.59073, 0.59184, 0.59040, 0.59184]

    broken_explore, broken_conv = replay(scores, eps=0.002, conv_n=3, explore_after=3)
    assert broken_conv == 7 and broken_explore is None, (
        "the historical config must reproduce the bug, or this test proves nothing"
    )

    fixed_explore, fixed_conv = replay(scores, eps=0.002, conv_n=4, explore_after=2)
    assert fixed_explore is not None
    assert fixed_conv is None or fixed_explore < fixed_conv


@pytest.mark.parametrize("conv_n", [2, 3, 4, 5])
def test_convergence_always_implies_explore_was_reachable_first(conv_n):
    """`explore_after < conv_n` is sufficient, for any trajectory.

    `best_history` is monotone, so `converged()` can only fire once each of the
    last conv_n improvements was <= eps — i.e. flat_iters >= conv_n by then.
    Anything strictly smaller must therefore have fired earlier.
    """
    import itertools

    eps, explore_after = 0.002, conv_n - 1
    steps = (0.0, 0.001, 0.05)  # flat-by-zero, flat-under-eps, a real jump
    for combo in itertools.product(steps, repeat=6):
        scores = [0.5]
        for d in combo:
            scores.append(scores[-1] + d)
        explore, conv = replay(scores, eps=eps, conv_n=conv_n, explore_after=explore_after)
        if conv is not None:
            assert explore is not None and explore < conv, (
                f"converged at {conv} without ever exploring; trajectory={scores}"
            )


def test_equal_explore_after_and_conv_n_is_the_bug():
    """Kept as the counter-example: equal values make explore unreachable."""
    flat_run = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    explore, conv = replay(flat_run, eps=0.002, conv_n=3, explore_after=3)
    assert conv is not None and explore is None
