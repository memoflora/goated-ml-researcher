"""Tests for the idea bank and the prompt text. Owner: D."""

from __future__ import annotations

import re
from pathlib import Path
from string import Template

import pytest

from orchestrator import knowledge
from orchestrator.contracts import Idea

PROMPTS = Path(__file__).resolve().parent.parent / "orchestrator" / "prompts"

# Exactly what agent.py substitutes. If B adds a variable, this list moves with it.
KNOWN_VARS = {
    "run_id", "iteration", "budget", "ideas", "history", "draft_angle",
    "parent_code", "parent_metrics", "parent_node_id", "error_class",
    "error_excerpt", "stdout_tail", "attempt", "max_attempts", "previous_attempts",
    # Task-derived, supplied by agent._task_values() to every prompt including system.md.
    # These are what let one set of prompt files serve more than one dataset.
    "dead_ends", "task_name", "task_kind", "task_description", "submission_header",
    "prediction_column", "group_column", "subsample_note", "order_note",
    "metric_names", "primary_expr",
}


# ------------------------------------------------------------------ the bank


def test_bank_loads_and_is_well_formed():
    ideas = knowledge.all_ideas()
    assert len(ideas) >= 30
    assert all(isinstance(i, Idea) for i in ideas)
    ids = [i.id for i in ideas]
    assert len(ids) == len(set(ids)), "duplicate idea ids"
    assert {i.tier for i in ideas} == {0, 1, 2, 3, 4}


def test_every_prerequisite_resolves():
    ideas = knowledge.all_ideas()
    ids = {i.id for i in ideas}
    for i in ideas:
        for p in i.prerequisites:
            assert p in ids, f"{i.id} requires unknown idea {p}"


def test_prerequisites_never_point_upward():
    """A prerequisite in a higher tier would deadlock: the gate can never open."""
    by_id = {i.id: i for i in knowledge.all_ideas()}
    for i in by_id.values():
        for p in i.prerequisites:
            assert by_id[p].tier <= i.tier, f"{i.id} (T{i.tier}) requires higher-tier {p}"


def test_tiering_matches_the_organisers_ranking():
    """The whole point of the retiering: losses before architecture.

    If someone 'fixes' this by moving DeepFM back to an early tier, the agent will spend
    its first real iterations on the direction measured least promising.
    """
    by_id = {i.id: i for i in knowledge.all_ideas()}
    assert by_id["T1.bpr-pairwise"].tier < by_id["T4.deepfm"].tier
    assert by_id["T1.listwise-softmax"].tier < by_id["T4.dcn-v2"].tier
    assert by_id["T2.mean-pooled-history"].tier < by_id["T4.deepfm"].tier


# ------------------------------------------------------------------ retrieval


def test_retrieve_starts_at_the_lowest_tier():
    got = knowledge.retrieve(tried=[], best_metrics={}, budget_left=50, k=5)
    assert got
    assert got[0].tier == 0


def test_retrieve_excludes_what_was_tried():
    first = knowledge.retrieve(tried=[], budget_left=50, k=5)
    tried = [first[0].id]
    again = knowledge.retrieve(tried=tried, budget_left=50, k=5)
    assert tried[0] not in {i.id for i in again}


def test_retrieve_gates_on_unmet_prerequisites():
    """Do not propose PLE before MMoE has been tried."""
    ids = {i.id for i in knowledge.retrieve(tried=[], budget_left=50, k=40)}
    assert "T3.ple" not in ids  # requires T3.mmoe
    with_mmoe = {
        i.id
        for i in knowledge.retrieve(
            tried=["T3.mmoe", "T3.aux-feedback-heads"],
            best_metrics={"primary": 0.61},
            budget_left=50,
            k=40,
        )
    }
    assert "T3.ple" in with_mmoe


def test_retrieve_fills_toward_k_when_asked_for_many():
    got = knowledge.retrieve(tried=[], best_metrics={"primary": 0.6}, budget_left=50, k=12)
    assert len(got) == 12, "a large k must fill, not stop at the working tier"


def test_retrieve_escalates_as_tiers_are_exhausted():
    tried = [i.id for i in knowledge.all_ideas() if i.tier == 0]
    got = knowledge.retrieve(tried=tried, budget_left=50, k=5)
    assert got and min(i.tier for i in got) == 1


def test_retrieve_includes_a_lookahead_from_the_next_tier():
    got = knowledge.retrieve(
        tried=["T0.reproduce-fm"], best_metrics={"primary": 0.60}, budget_left=50, k=5
    )
    assert len({i.tier for i in got}) > 1, "no lookahead: escalation would be a cliff"


def test_a_scored_pipeline_unlocks_tier_one_without_citing_the_idea():
    """The deadlock this guards: every T1+ idea sits behind T0.reproduce-fm. An agent
    that writes a perfect baseline but never cites that id would be offered tier 0
    forever, and the whole tiering would be inert for the run."""
    stuck = knowledge.retrieve(tried=[], best_metrics={}, budget_left=50, k=40)
    assert {i.tier for i in stuck} == {0}, "sanity: nothing scored, nothing unlocked"

    unlocked = knowledge.retrieve(
        tried=[], best_metrics={"primary": 0.6015}, budget_left=50, k=40
    )
    assert 1 in {i.tier for i in unlocked}
    assert "T1.bpr-pairwise" in {i.id for i in unlocked}


def test_retrieve_drops_slow_ideas_when_the_budget_is_nearly_spent():
    """An idea that cannot finish is worse than no idea - it burns the turn and leaves
    a broken pipeline behind."""
    got = knowledge.retrieve(tried=[], budget_left=2, k=8)
    assert got
    assert all(i.est_minutes <= knowledge.SHORT_BUDGET_MINUTES for i in got)


def test_retrieve_respects_k():
    for k in (1, 3, 5):
        assert len(knowledge.retrieve(tried=[], budget_left=50, k=k)) <= k
    assert knowledge.retrieve(tried=[], budget_left=50, k=0) == []


def test_retrieve_is_deterministic():
    a = [i.id for i in knowledge.retrieve(tried=["T0.reproduce-fm"], budget_left=30, k=5)]
    b = [i.id for i in knowledge.retrieve(tried=["T0.reproduce-fm"], budget_left=30, k=5)]
    assert a == b


def test_retrieve_never_raises_on_a_broken_bank(tmp_path: Path):
    """core.py catches this, but an exception still costs the iteration its ideas."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("ideas: [{id: x}]\n", encoding="utf-8")  # missing required keys
    assert knowledge.retrieve(tried=[], budget_left=50, k=5, path=str(bad)) == []


def test_exhausted_bank_returns_empty_not_an_error():
    everything = [i.id for i in knowledge.all_ideas()]
    assert knowledge.retrieve(tried=everything, budget_left=50, k=5) == []


# ------------------------------------------------------------------ prompts


@pytest.mark.parametrize("name", ["system", "draft", "improve", "repair"])
def test_prompt_exists_and_is_substantive(name: str):
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    assert len(text.split()) > 120, f"{name}.md is too thin to be doing any work"


@pytest.mark.parametrize("name", ["system", "draft", "improve", "repair"])
def test_prompt_uses_only_variables_agent_supplies(name: str):
    """A typo'd variable silently renders as a literal, which is invisible until a run
    produces a prompt with a dangling placeholder in it."""
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    used = set(re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", text))
    assert used <= KNOWN_VARS, f"{name}.md uses unknown variables: {used - KNOWN_VARS}"


@pytest.mark.parametrize(
    "name,required",
    [
        ("draft", {"draft_angle", "ideas", "budget"}),
        ("improve", {"parent_code", "parent_metrics", "history", "ideas"}),
        ("repair", {"parent_code", "error_class", "error_excerpt", "attempt"}),
    ],
)
def test_prompt_renders_the_variables_it_needs(name: str, required: set[str]):
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    used = set(re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", text))
    assert required <= used, f"{name}.md never renders {required - used}"


@pytest.mark.parametrize("name", ["system", "draft", "improve", "repair"])
def test_prompt_substitutes_cleanly(name: str):
    """safe_substitute must leave no placeholder behind once every known var is given."""
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    out = Template(text).safe_substitute({v: "X" for v in KNOWN_VARS})
    assert "$" not in out, f"{name}.md has an unsubstituted placeholder"


def test_system_prompt_states_the_disqualifying_rules():
    text = (PROMPTS / "system.md").read_text(encoding="utf-8").lower()
    assert "no external training data" in text
    assert "test" in text and "validation" in text
    assert "submission.csv" in text and "result_json" in text
    assert "row_id" in text


def test_system_prompt_carries_every_dead_end():
    """The dead ends are static, so they ride in the cached system block rather than
    being re-sent each call.

    They used to be typed into `system.md` by hand, and this test compared the two files.
    They are now injected from the *task's own* bank, because a second task must not
    inherit KuaiRand's measured conclusions. So the assertion moved to the rendered
    prompt: whatever the bank says must reach the model.
    """
    from string import Template

    ends = knowledge.dead_ends()
    assert len(ends) >= 4

    raw = (PROMPTS / "system.md").read_text(encoding="utf-8")
    assert "$dead_ends" in raw, "system.md no longer injects the bank's dead ends"

    rendered = Template(raw).safe_substitute(
        {"dead_ends": "\n\n".join(f"- **{d.claim}** {d.verdict}" for d in ends)}
    ).lower()

    # each dead end must be recognisable by its distinguishing number/idea
    assert "0.5940" in rendered and "0.5950" in rendered   # more static features
    assert "0.5895" in rendered or "0.5902" in rendered    # more capacity
    assert "within-user" in rendered                       # user-side first-order terms
    assert "0.5982" in rendered and "0.6032" in rendered   # uniform vs weighted sampling
    assert "positive count" in rendered


def test_a_task_never_inherits_another_tasks_dead_ends():
    """The failure this guards against is silent and expensive: a regression task told,
    with authority, that a larger embedding dimension does not help."""
    from orchestrator.taskspec import load_task

    generic = load_task("demo-regression")
    assert generic.ideas_path is not None
    assert knowledge.dead_ends(str(generic.ideas_path)) == []

    kuairand = load_task("kuairand-pure")
    assert len(knowledge.dead_ends(str(kuairand.ideas_path))) >= 4


def test_improve_prompt_demands_one_change_and_warns_about_noise():
    text = (PROMPTS / "improve.md").read_text(encoding="utf-8").lower()
    assert "one focused change" in text
    assert "0.0008" in text, "the agent must know what noise looks like"


def test_repair_prompt_forbids_scope_creep():
    text = (PROMPTS / "repair.md").read_text(encoding="utf-8").lower()
    assert "change nothing else" in text
    for cls in ("syntax", "import", "contract", "timeout", "oom"):
        assert cls in text, f"repair.md gives no guidance for error_class={cls}"
