"""The data card is the agent's only view of the data, so its budget and its
facts-only discipline are both enforced here."""

from __future__ import annotations

import re

import pytest

from orchestrator.datacard import data_card, estimate_tokens
from orchestrator.splits import DEFAULT_DATA_DIR

needs_data = pytest.mark.skipif(
    not DEFAULT_DATA_DIR.is_dir(), reason="KuaiRand-Pure not downloaded"
)

pytestmark = needs_data


@pytest.fixture(scope="module")
def card() -> str:
    return data_card()


def test_fits_the_token_budget(card):
    assert estimate_tokens(card) <= 3000, f"data card is ~{estimate_tokens(card)} tokens"


def test_is_deterministic(card):
    assert data_card() == card


def test_states_the_published_composition_numbers(card):
    # These are the numbers the organisers published; if our computation drifts from
    # them the card is lying to the agent.
    assert "27.1%" in card  # zero-positive test users
    assert "9.2%" in card  # all-positive test users
    assert "3.06%" in card  # repeated (user_id, video_id) test rows
    assert "0.8645" in card  # attainable ceiling


def test_lists_split_sizes(card):
    for n in ("1,141,112", "124,909", "170,588"):
        assert n in card


def test_covers_every_feedback_signal(card):
    from orchestrator.datacard import BINARY_SIGNALS, CONTINUOUS_SIGNALS

    for sig in BINARY_SIGNALS + CONTINUOUS_SIGNALS:
        assert f"`{sig}`" in card, f"{sig} missing from the data card"


def test_states_the_leakage_constraints(card):
    lowered = card.lower()
    assert "no external training data" in lowered
    assert "out-of-fold" in lowered
    assert "users" in lowered and "not rows" in lowered  # the subsample trap


def test_contains_no_advice(card):
    """Facts only. Recommendations belong in the idea bank so they stay attributable."""
    banned = [
        r"\byou should\b",
        r"\bwe recommend\b",
        r"\btry \b",
        r"\bconsider \b",
        r"\bit is best\b",
        r"\bthe best approach\b",
        r"\bshould be used\b",
    ]
    hits = [p for p in banned if re.search(p, card, flags=re.I)]
    assert not hits, f"advice leaked into the data card: {hits}"
