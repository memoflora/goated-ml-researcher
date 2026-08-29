"""RESULTS.md must be generatable from a journal alone, at any point, including a
truncated or partly-corrupt one — it is the deliverable that has to exist from H+60."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orchestrator import report

FIXTURE = Path(__file__).parent / "fixtures" / "journal_sample.jsonl"


@pytest.fixture
def run_dir(tmp_path) -> Path:
    d = tmp_path / "r20260830-0412"
    d.mkdir()
    shutil.copy(FIXTURE, d / "journal.jsonl")
    return d


@pytest.fixture
def summary():
    return report.summarise(report.read_journal(FIXTURE))


def test_survives_a_malformed_line(summary):
    assert summary.malformed_lines == 1
    assert summary.scored_nodes == 3  # and still parses everything else


def test_picks_the_best_node(summary):
    assert summary.best["node_id"] == "n004"
    assert summary.best["metrics"]["primary"] == pytest.approx(0.6161)


def test_counts_resources(summary):
    assert summary.tokens_in == 44_512
    assert summary.tokens_out == 12_290
    assert summary.iterations == 5
    assert summary.converged is True


def test_counts_robustness_events(summary):
    assert sum(summary.error_classes.values()) == 2
    assert summary.error_classes["syntax"] == 1
    assert summary.error_classes["oom"] == 1
    assert summary.recoveries == 2
    assert summary.dead_nodes == 1  # the route_around


def test_intervention_count_is_zero_for_a_clean_run(summary):
    assert summary.interventions == 0


def test_counts_interventions_when_present(tmp_path):
    j = tmp_path / "journal.jsonl"
    j.write_text(
        json.dumps({"run_id": "r1", "event": "intervention", "iteration": 3}) + "\n"
        + json.dumps({"run_id": "r1", "event": "intervention", "iteration": 7}) + "\n"
    )
    s = report.summarise(report.read_journal(j))
    assert s.interventions == 2


def test_falls_back_to_trajectory_best_without_best_updated(tmp_path):
    j = tmp_path / "journal.jsonl"
    j.write_text(
        "\n".join(
            json.dumps(
                {
                    "run_id": "r1",
                    "event": "eval",
                    "iteration": i,
                    "node_id": f"n00{i}",
                    "metrics": {"gauc": 0.6, "ndcg@5": 0.5, "primary": p},
                }
            )
            for i, p in enumerate([0.60, 0.63, 0.61], start=1)
        )
        + "\n"
    )
    s = report.summarise(report.read_journal(j))
    assert s.best["node_id"] == "n002"


def test_renders_the_absolute_delta_over_baseline(run_dir):
    report.build(run_dir)
    md = (run_dir / "RESULTS.md").read_text()
    assert "absolute delta" in md
    assert "+0.0145" in md  # primary 0.6161 - 0.6016
    assert "Manual interventions during this run: 0" in md
    assert "Total LLM tokens" in md


def test_records_every_hypothesis(run_dir):
    report.build(run_dir)
    md = (run_dir / "RESULTS.md").read_text()
    # including the node that died -- what it tried still counts for Innovation
    assert "n003" in md
    assert "out-of-fold smoothed item CTR priors" in md
    assert "T4.listwise-softmax" in md


def test_writes_trajectory_png(run_dir):
    report.build(run_dir)
    png = run_dir / "trajectory.png"
    pytest.importorskip("matplotlib")
    assert png.is_file() and png.stat().st_size > 1000


def test_empty_journal_still_renders(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "journal.jsonl").write_text("")
    s = report.build(d)
    assert s.best is None
    assert "No scored node" in (d / "RESULTS.md").read_text()


def test_missing_journal_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.build(tmp_path)
