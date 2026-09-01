"""The per-iteration run log — Starter Kit Run-log requirement (§2.5 item 3).

Four things are required per iteration: the hypothesis, the code diff applied, the
resulting metrics, and any error/recovery. Three come straight from the journal; the
diff does not exist anywhere until this module reconstructs it, which is why it is
tested rather than eyeballed.
"""

from __future__ import annotations

import json

import pytest

from orchestrator import runlog

EMPTY_INTERVENTIONS = """\
# Manual interventions — rtest

Every human touch during this run, timestamped. The count is directly
scored under Autonomy. An empty table is the goal.

| UTC | Who | What they did | Why the agent could not |
|---|---|---|---|
"""

PARENT_CODE = "import pandas as pd\n\n\ndef fit(x):\n    return x.mean()\n"
CHILD_CODE = "import pandas as pd\n\n\ndef fit(x):\n    return x.median()\n"


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "rtest"
    (d / "nodes" / "n000").mkdir(parents=True)
    (d / "nodes" / "n001").mkdir(parents=True)
    (d / "nodes" / "n000" / "pipeline.py").write_text(PARENT_CODE)
    (d / "nodes" / "n001" / "pipeline.py").write_text(CHILD_CODE)
    (d / "interventions.md").write_text(EMPTY_INTERVENTIONS)
    (d / "state.json").write_text(json.dumps({"tree": {"nodes": {
        "n000": {"parent_id": None},
        "n001": {"parent_id": "n000"},
    }}}))
    events = [
        {"event": "run_start", "iteration": 0, "run_id": "rtest", "task": "kuairand-pure",
         "mode": "dev", "git_sha": "abc123", "max_iters": 8, "wall_clock_s": 1800,
         "conv_eps": 0.002, "conv_n": 4, "explore_after": 2, "subsample": 1.0},
        {"event": "proposal", "iteration": 1, "node_id": "n000", "kind": "draft",
         "model": "gpt-5.6-terra", "hypothesis": "Reproduce the official FM baseline.",
         "plan": ["load data", "fit FM"], "idea_ids": ["T0.reproduce-fm"]},
        {"event": "eval", "iteration": 1, "node_id": "n000",
         "metrics": {"gauc": 0.6255, "ndcg@5": 0.5175, "primary": 0.5715},
         "delta_vs_baseline": {"gauc": -0.0419, "ndcg@5": -0.0182, "primary": -0.0301}},
        {"event": "proposal", "iteration": 2, "node_id": "n001", "kind": "improve",
         "model": "gpt-5.6-terra", "hypothesis": "A median is robust to outliers.",
         "plan": ["swap mean for median"], "idea_ids": []},
        {"event": "error", "iteration": 2, "node_id": "n001",
         "error_class": "runtime", "error_excerpt": "ValueError: shapes not aligned"},
        {"event": "recovery", "iteration": 2, "node_id": "n001",
         "recovery": "schedule_debug", "repair_attempt": 1, "max_repairs": 3},
    ]
    (d / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return d


class TestInterventionCount:
    def test_an_empty_log_counts_zero(self, run_dir):
        """The bug this pins: the file is a table with a prose preamble, so counting
        non-empty lines reports 4 for a run with no interventions at all. Autonomy is
        scored directly on this number."""
        assert runlog.count_interventions(run_dir) == (0, "")

    def test_real_interventions_are_counted(self, run_dir):
        (run_dir / "interventions.md").write_text(
            EMPTY_INTERVENTIONS
            + "| 02:14 | ops | restarted the run | the harness deadlocked |\n"
            + "| 02:51 | ops | fixed a path | wrong data dir |\n"
        )
        assert runlog.count_interventions(run_dir)[0] == 2

    def test_a_missing_file_is_reported_not_guessed(self, tmp_path):
        n, note = runlog.count_interventions(tmp_path)
        assert n == 0 and "interventions.md" in note


class TestDiff:
    def test_an_improve_diffs_against_its_parent(self, run_dir):
        lines, note = runlog.diff_lines(run_dir, "n001", "n000")
        assert note == ""
        body = "".join(lines)
        assert "-    return x.mean()" in body
        assert "+    return x.median()" in body

    def test_a_draft_says_so_rather_than_dumping_the_file(self, run_dir):
        """A draft has no parent. Emitting the whole new file would bury the log."""
        lines, note = runlog.diff_lines(run_dir, "n000", None)
        assert lines == []
        assert "no parent" in note and "5 lines" in note

    def test_an_unchanged_pipeline_is_stated_explicitly(self, run_dir):
        (run_dir / "nodes" / "n001" / "pipeline.py").write_text(PARENT_CODE)
        _, note = runlog.diff_lines(run_dir, "n001", "n000")
        assert "no change" in note

    def test_a_missing_pipeline_does_not_raise(self, run_dir):
        _, note = runlog.diff_lines(run_dir, "n999", "n000")
        assert "no `pipeline.py`" in note


class TestRender:
    def test_every_required_element_is_present(self, run_dir):
        md = runlog.render(run_dir)
        assert "Reproduce the official FM baseline." in md      # hypothesis
        assert "```diff" in md and "x.median()" in md           # code diff
        assert "0.62551" in md or "0.6255" in md                # metrics
        assert "gauc" in md and "ndcg@5" in md                  # the graded metrics
        assert "runtime" in md and "shapes not aligned" in md   # error
        assert "schedule_debug" in md                           # recovery
        assert "**0**" in md                                    # interventions

    def test_the_model_and_search_limits_are_recorded(self, run_dir):
        md = runlog.render(run_dir)
        assert "gpt-5.6-terra" in md
        assert "explore_after=2" in md and "conv_n=4" in md

    def test_a_long_diff_is_truncated_with_a_pointer(self, run_dir):
        (run_dir / "nodes" / "n001" / "pipeline.py").write_text(
            "".join(f"x = {i}\n" for i in range(400)))
        md = runlog.render(run_dir, max_diff_lines=20)
        assert "diff truncated at 20" in md
        assert "nodes/n001/pipeline.py" in md

    def test_a_torn_journal_line_does_not_lose_the_run(self, run_dir):
        with (run_dir / "journal.jsonl").open("a") as fh:
            fh.write('{"event": "proposal", "iterat')   # killed mid-write
        assert "Reproduce the official FM baseline." in runlog.render(run_dir)

    def test_it_works_without_a_checkpoint(self, run_dir):
        """A run killed before its first checkpoint still has a journal worth reading."""
        (run_dir / "state.json").unlink()
        md = runlog.render(run_dir)
        assert "Reproduce the official FM baseline." in md
