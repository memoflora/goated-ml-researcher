"""Tests for the scoring seam.

The rejection tests build tiny submissions against a synthetic 6-row split so they run
without the dataset. The reproduction tests need the real data and skip without it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from orchestrator import evaluate as ev
from orchestrator.splits import DEFAULT_DATA_DIR, Split, normalise_split

HAVE_DATA = DEFAULT_DATA_DIR.is_dir()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="KuaiRand-Pure not downloaded")

# Published reference numbers (vendor/starter_kit/baseline_scores.json).
BASELINE_VAL = {"gauc": 0.6674, "ndcg@5": 0.5357, "primary": 0.6016}


@pytest.fixture
def fake_split() -> Split:
    return Split(
        name="valid",
        user_ids=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        video_ids=np.array([10, 11, 12, 10, 13, 14], dtype=np.int64),
        labels=np.array([1, 0, 0, 0, 1, 1], dtype=np.int8),
    )


def write_csv(path: Path, rows: list[list], header: list[str] | None = None) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header if header is not None else ev.HEADER)
        w.writerows(rows)
    return path


def good_rows(split: Split) -> list[list]:
    return [
        [i, int(u), int(v), f"{0.5 - 0.1 * i:.4f}"]
        for i, (u, v) in enumerate(zip(split.user_ids, split.video_ids))
    ]


def test_normalise_split_accepts_val_and_valid():
    assert normalise_split("val") == "valid"
    assert normalise_split("valid") == "valid"
    assert normalise_split("TEST") == "test"
    with pytest.raises(ValueError):
        normalise_split("holdout")


def test_accepts_a_well_formed_submission(tmp_path, fake_split):
    p = write_csv(tmp_path / "s.csv", good_rows(fake_split))
    ev._read_scores(p, fake_split)  # must not raise


@pytest.mark.parametrize(
    "mutate, expect",
    [
        pytest.param(lambda r: r, "", id="control"),
        pytest.param(lambda r: r[:-1], "rows", id="too-few-rows"),
        pytest.param(lambda r: r + [[6, 2, 15, "0.1"]], "more rows", id="too-many-rows"),
        pytest.param(
            lambda r: [x if i != 3 else [99, x[1], x[2], x[3]] for i, x in enumerate(r)],
            "row_id",
            id="row-id-gap",
        ),
        pytest.param(
            lambda r: [x if i != 2 else [x[0], 999, x[2], x[3]] for i, x in enumerate(r)],
            "misaligned",
            id="user-misaligned",
        ),
        pytest.param(
            lambda r: [x if i != 2 else [x[0], x[1], 999, x[3]] for i, x in enumerate(r)],
            "misaligned",
            id="video-misaligned",
        ),
        pytest.param(
            lambda r: [x if i != 1 else [x[0], x[1], x[2], "abc"] for i, x in enumerate(r)],
            "not a number",
            id="non-numeric-score",
        ),
        pytest.param(
            lambda r: [x if i != 1 else [x[0], x[1], x[2], "nan"] for i, x in enumerate(r)],
            "NaN/Inf",
            id="nan-score",
        ),
        pytest.param(
            lambda r: [x if i != 1 else [x[0], x[1], x[2], "inf"] for i, x in enumerate(r)],
            "NaN/Inf",
            id="inf-score",
        ),
        pytest.param(
            lambda r: [x if i != 1 else [x[0], x[1], x[2]] for i, x in enumerate(r)],
            "fields",
            id="wrong-field-count",
        ),
    ],
)
def test_rejection_cases(tmp_path, fake_split, mutate, expect):
    rows = mutate(good_rows(fake_split))
    p = write_csv(tmp_path / "s.csv", rows)
    if not expect:  # the control must pass
        ev._read_scores(p, fake_split)
        return
    with pytest.raises(ev.SubmissionError) as exc:
        ev._read_scores(p, fake_split)
    assert expect in str(exc.value)


def test_rejects_wrong_header(tmp_path, fake_split):
    p = write_csv(
        tmp_path / "s.csv", good_rows(fake_split), header=["row_id", "user_id", "vid", "score"]
    )
    with pytest.raises(ev.SubmissionError, match="header"):
        ev._read_scores(p, fake_split)


def test_score_arrays_matches_hand_computed(fake_split):
    """Two users, each with a perfect ranking -> GAUC 1.0, nDCG 1.0."""
    perfect = fake_split.labels.astype(float)
    m = ev.score_arrays(fake_split.user_ids, fake_split.labels, perfect)
    assert m["gauc"] == pytest.approx(1.0)
    assert m["ndcg@5"] == pytest.approx(1.0)
    assert m["primary"] == pytest.approx(1.0)


def test_score_arrays_zero_positive_user_counts_as_ndcg_zero():
    """A user with no positive contributes nDCG 0 and is excluded from GAUC."""
    users = np.array([1, 1, 2, 2], dtype=np.int64)
    labels = np.array([1, 0, 0, 0], dtype=np.int8)
    m = ev.score_arrays(users, labels, np.array([1.0, 0.0, 1.0, 0.0]))
    assert m["gauc"] == pytest.approx(1.0)  # only user 1 counts
    assert m["ndcg@5"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2


def test_delta_vs_baseline():
    m = {"gauc": 0.6774, "ndcg@5": 0.5457, "primary": 0.6116}
    d = ev.delta_vs_baseline(m, BASELINE_VAL)
    assert d["primary"] == pytest.approx(0.0100, abs=1e-6)
    assert d["gauc"] == pytest.approx(0.0100, abs=1e-6)


# --------------------------------------------------------------------------- with real data


@needs_data
def test_validate_reports_missing_file():
    ok, msg = ev.validate(Path("/nonexistent/submission.csv"), "val")
    assert not ok and "not found" in msg


@needs_data
def test_hidden_test_scoring_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(ev.TEST_SCORING_ENV, raising=False)
    with pytest.raises(ev.SubmissionError, match="refusing to score the hidden test"):
        ev.score(tmp_path / "anything.csv", "test")


@needs_data
def test_split_sizes_match_the_official_numbers():
    from orchestrator.splits import load_splits

    sizes = {k: len(v) for k, v in load_splits().items()}
    assert sizes == {"train": 1_141_112, "valid": 124_909, "test": 170_588}


@needs_data
def test_item_popularity_rung_reproduces():
    """The published sanity rung: item popularity -> validation primary 0.5807.

    If this drifts, the harness is wrong and every metric we report is wrong with it.
    """
    import collections

    from orchestrator.splits import get_split

    tr, va = get_split("train"), get_split("valid")
    pos, imp = collections.Counter(), collections.Counter()
    for v, y in zip(tr.video_ids.tolist(), tr.labels.tolist()):
        imp[v] += 1
        pos[v] += y
    gmean = sum(pos.values()) / sum(imp.values())
    prior = 20.0
    scores = np.array(
        [
            (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
            for v in va.video_ids.tolist()
        ]
    )
    m = ev.score_arrays(va.user_ids, va.labels, scores)
    assert m["primary"] == pytest.approx(0.5807, abs=0.0005)
    assert m["gauc"] == pytest.approx(0.6387, abs=0.0005)
    assert m["ndcg@5"] == pytest.approx(0.5227, abs=0.0005)
