"""Task configuration, metrics and the generic data path.

These cover the seam that makes the orchestrator dataset-agnostic. The failures they guard
against are all of one kind: they do not raise, they silently do the KuaiRand thing on a
task that is not KuaiRand, and the run completes looking healthy.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from orchestrator import datasource as ds
from orchestrator import metrics as M
from orchestrator import taskspec as ts

# --------------------------------------------------------------------------- metrics


def test_every_registered_metric_declares_a_direction():
    for name in M.available():
        assert isinstance(M.get(name).greater_is_better, bool)


def test_primary_negates_lower_is_better_metrics():
    """`core.py` only ever maximises, so the sign correction has to happen here."""
    assert M.primary_of({"rmse": 2.0}, ("rmse",)) == -2.0
    assert M.primary_of({"r2": 0.8}, ("r2",)) == 0.8
    # 0.60155 exactly; the organisers publish it rounded to 0.6016.
    assert M.primary_of({"gauc": 0.6674, "ndcg@5": 0.5357}, ("gauc", "ndcg@5")) == pytest.approx(
        0.6016, abs=5e-5
    )


def test_parametric_metrics_accept_any_k():
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    g = np.array([0, 0, 0, 1, 1, 1])
    for name in ("ndcg@1", "ndcg@10", "map@3", "recall@2"):
        assert 0.0 <= M.compute(name, y, p, g) <= 1.0


def test_unknown_metric_names_fail_loudly():
    with pytest.raises(KeyError):
        M.get("definitely_not_a_metric")


def test_grouped_metrics_refuse_to_run_without_groups():
    y, p = np.array([1, 0]), np.array([0.6, 0.4])
    with pytest.raises(ValueError, match="group"):
        M.compute("gauc", y, p, None)


def test_perfect_and_inverted_rankings_bracket_the_scale():
    y = np.array([1, 0, 1, 0])
    g = np.array([0, 0, 1, 1])
    assert M.compute("gauc", y, np.array([1.0, 0.0, 1.0, 0.0]), g) == 1.0
    assert M.compute("gauc", y, np.array([0.0, 1.0, 0.0, 1.0]), g) == 0.0


def test_rmse_and_mae_are_zero_on_a_perfect_fit():
    y = np.array([1.0, 2.0, 3.0])
    assert M.compute("rmse", y, y) == 0.0
    assert M.compute("mae", y, y) == 0.0


# ----------------------------------------------------------------------- task config


MINIMAL = {
    "name": "t",
    "kind": "regression",
    "description": "Predict a number from some columns.",
    "data": {"dir": "data", "file": "x.csv", "target": "y"},
    "metrics": {"primary": ["rmse"]},
}


def test_a_minimal_task_parses():
    t = ts.parse_task(dict(MINIMAL))
    assert t.primary_parts == ("rmse",)
    assert t.submission_columns[0] == "row_id"
    assert t.prediction_column in t.submission_columns


def test_description_is_required():
    """It is the problem statement; without it the agent is writing code blind."""
    raw = dict(MINIMAL, description="")
    with pytest.raises(ts.TaskConfigError, match="description"):
        ts.parse_task(raw)


def test_an_unknown_metric_is_rejected_at_parse_time():
    raw = dict(MINIMAL, metrics={"primary": ["not_a_metric"]})
    with pytest.raises(ts.TaskConfigError, match="not_a_metric"):
        ts.parse_task(raw)


def test_a_grouped_metric_without_a_group_column_is_rejected():
    raw = dict(MINIMAL, kind="ranking", metrics={"primary": ["gauc"]})
    with pytest.raises(ts.TaskConfigError, match="group"):
        ts.parse_task(raw)


def test_submission_columns_must_contain_row_id():
    raw = dict(MINIMAL, submission={"columns": ["id", "prediction"]})
    with pytest.raises(ts.TaskConfigError, match="row_id"):
        ts.parse_task(raw)


def test_date_split_needs_a_column_and_ranges():
    raw = dict(MINIMAL)
    raw["data"] = dict(raw["data"], split={"strategy": "date"})
    with pytest.raises(ts.TaskConfigError, match="date_column"):
        ts.parse_task(raw)


def test_explore_after_defaults_below_conv_n():
    """The policy's explore branch is unreachable unless this holds."""
    t = ts.parse_task(dict(MINIMAL))
    assert t.explore_after < t.conv_n


def test_explore_after_and_conv_n_parse_from_limits():
    raw = dict(MINIMAL, limits={"conv_n": 6, "explore_after": 3})
    t = ts.parse_task(raw)
    assert (t.conv_n, t.explore_after) == (6, 3)


@pytest.mark.parametrize("explore_after,conv_n", [(3, 3), (4, 3), (5, 2)])
def test_explore_after_at_or_above_conv_n_is_rejected(explore_after, conv_n):
    """The bug that cost a live run 37 iterations, now unrepresentable.

    With explore_after >= conv_n the run stops on exactly the iteration the
    explore branch first becomes reachable, so a plateau ends the search instead
    of redirecting it. A task file may no longer express that.
    """
    raw = dict(MINIMAL, limits={"conv_n": conv_n, "explore_after": explore_after})
    with pytest.raises(ts.TaskConfigError, match="explore_after"):
        ts.parse_task(raw)


def test_a_task_without_its_own_bank_gets_the_domain_free_default():
    """Inheriting another dataset's idea bank means inheriting its conclusions."""
    t = ts.parse_task(dict(MINIMAL))
    assert t.ideas_path == ts.DEFAULT_IDEAS


# ------------------------------------------------------------- the shipped task files


def test_the_shipped_tasks_all_parse():
    names = ts.available_tasks()
    assert "kuairand-pure" in names
    for name in names:
        ts.load_task(name)


def test_every_shipped_task_can_actually_explore():
    """Guards the whole `tasks/` directory, not just the one we edited."""
    for name in ts.available_tasks():
        t = ts.load_task(name)
        assert t.explore_after < t.conv_n, (
            f"{name}: explore_after={t.explore_after} >= conv_n={t.conv_n}, so the "
            "policy's explore branch can never run"
        )


def test_kuairand_task_matches_the_published_numbers():
    t = ts.load_task("kuairand-pure")
    assert t.baseline_val["primary"] == pytest.approx(0.6016)
    assert t.ceiling == pytest.approx(0.8645)
    assert t.primary({"gauc": 0.6674, "ndcg@5": 0.5357}) == pytest.approx(0.6016, abs=5e-5)
    assert t.submission_columns == ("row_id", "user_id", "video_id", "score")


# ------------------------------------------------------------------- generic loading


@pytest.fixture
def tiny_task(tmp_path):
    rows = [{"id": i, "a": i % 5, "b": i * 0.5, "y": i * 2.0} for i in range(100)]
    path = tmp_path / "all.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "a", "b", "y"])
        w.writeheader()
        w.writerows(rows)
    return ts.parse_task(
        {
            "name": "tiny",
            "kind": "regression",
            "description": "Predict y from a and b.",
            "data": {
                "dir": str(tmp_path),
                "file": "all.csv",
                "target": "y",
                "id_columns": ["id"],
                "split": {"strategy": "random", "valid_frac": 0.2, "test_frac": 0.2, "seed": 0},
            },
            "submission": {"columns": ["row_id", "id", "prediction"]},
            "metrics": {"primary": ["rmse"], "report": ["rmse", "mae"]},
        }
    )


def test_random_split_partitions_every_row_exactly_once(tiny_task):
    sizes = {n: len(ds.eval_arrays(tiny_task, n)) for n in ds.split_names(tiny_task)}
    assert sum(sizes.values()) == 100
    assert sizes["valid"] == 20 and sizes["test"] == 20


def test_splits_are_deterministic(tiny_task):
    first = ds.eval_arrays(tiny_task, "valid").ids["id"].tolist()
    ds._build_cache(tiny_task)  # force a rebuild rather than a cache read
    assert ds.eval_arrays(tiny_task, "valid").ids["id"].tolist() == first


def test_materialise_strips_the_target_from_the_test_split(tiny_task):
    """The no-peeking rule should hold because the labels are absent, not because we
    asked the agent not to look at them."""
    out = ds.materialise(tiny_task)
    with open(out / "test.csv", encoding="utf-8") as fh:
        assert "y" not in next(csv.reader(fh))
    with open(out / "train.csv", encoding="utf-8") as fh:
        assert "y" in next(csv.reader(fh))


def test_group_split_never_splits_a_group(tmp_path):
    rows = [{"g": i // 10, "x": i, "y": float(i)} for i in range(100)]
    path = tmp_path / "g.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["g", "x", "y"])
        w.writeheader()
        w.writerows(rows)
    task = ts.parse_task(
        {
            "name": "grp",
            "kind": "regression",
            "description": "Predict y, grouped by g.",
            "data": {
                "dir": str(tmp_path),
                "file": "g.csv",
                "target": "y",
                "group": "g",
                "split": {"strategy": "group", "valid_frac": 0.3, "seed": 1},
            },
            "metrics": {"primary": ["rmse"]},
        }
    )
    frames = ds.all_frames(task)
    train_g = set(frames["train"]["g"])
    valid_g = set(frames["valid"]["g"])
    assert not (train_g & valid_g), "a group appeared in both splits"


# ---------------------------------------------------------------- generic evaluation


def _write_submission(path, task, arrays, values):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(task.submission_columns)
        for i, v in enumerate(values):
            w.writerow([i, arrays.ids["id"][i], v])


def test_generic_score_and_validate_round_trip(tiny_task, tmp_path):
    from orchestrator import evaluate as ev

    arrays = ds.eval_arrays(tiny_task, "valid")
    sub = tmp_path / "s.csv"
    _write_submission(sub, tiny_task, arrays, arrays.target)  # a perfect prediction

    ok, msg = ev.validate(sub, "valid", tiny_task)
    assert ok, msg
    scored = ev.score(sub, "valid", tiny_task)
    assert scored["rmse"] == pytest.approx(0.0)
    assert scored["primary"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda rows: rows[:-1], "rows"),                                 # one row short
        (lambda rows: [["wrong", "header", "here"], *rows[1:]], "header"),  # wrong header
        (lambda rows: [rows[0], [99, rows[1][1], 1.0], *rows[2:]], "row_id"),  # id gap
        (lambda rows: [rows[0], [0, 123456789, 1.0], *rows[2:]], "misaligned"),  # bad id
        (lambda rows: [rows[0], [0, rows[1][1], "nope"], *rows[2:]], "not a number"),
        (lambda rows: [rows[0], [0, rows[1][1], float("nan")], *rows[2:]], "NaN"),
    ],
)
def test_generic_validate_rejects_malformed_submissions(tiny_task, tmp_path, mutate, expected):
    from orchestrator import evaluate as ev

    arrays = ds.eval_arrays(tiny_task, "valid")
    rows = [list(tiny_task.submission_columns)]
    rows += [[i, arrays.ids["id"][i], 1.0] for i in range(len(arrays))]

    sub = tmp_path / "bad.csv"
    with open(sub, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(mutate(rows))

    ok, msg = ev.validate(sub, "valid", tiny_task)
    assert not ok
    assert expected.lower() in msg.lower(), msg
