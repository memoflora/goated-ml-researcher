"""Guards on the artifacts the organisers grade, not on the code that makes them.

Problem statement §2.5 requires the run logs (3), the manual-intervention count (3b)
and the final submission (4a). All three exist only inside a run directory, and `runs/`
is gitignored — so they reach the public repository only by being copied into
`runs/examples/`. These tests guard that seam, because it fails silently: the files sit
on disk looking committed, and are simply absent from the clone a judge opens.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.archive_run import TOP_LEVEL, archive

REPO_ROOT = Path(__file__).resolve().parents[1]


def ignored(path: Path) -> bool:
    """Would git leave this file out of a clone?"""
    return subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=REPO_ROOT, capture_output=True,
    ).returncode == 0


def make_run(root: Path) -> Path:
    """A run directory shaped like a real one, including the parts we do NOT want."""
    run = root / "r20260831-9999"
    (run / "final").mkdir(parents=True)
    (run / "best").mkdir(parents=True)
    for name in TOP_LEVEL:
        (run / name).write_text("{}\n", encoding="utf-8")
    (run / "final" / "submission.csv").write_text("row_id,user_id,video_id,score\n0,1,2,0.5\n")
    (run / "best" / "submission.csv").write_text("row_id,user_id,video_id,score\n0,1,2,0.5\n")
    (run / "best" / "pipeline.py").write_text("# winner\n")
    for i in range(3):
        node = run / "nodes" / f"n{i:03d}"
        node.mkdir(parents=True)
        (node / "pipeline.py").write_text(f"# node {i}\n")
        # The bulk a real run carries, and the reason this script is selective:
        # 170,588 rows per node, reproducible, evidence of nothing.
        (node / "submission.csv").write_text("row_id\n" + "0\n" * 100)
        (node / "stdout.log").write_text("x" * 10_000)
    return run


class TestGitignoreExemption:
    """The rule that actually puts the deliverables in the repository."""

    def test_a_submission_under_examples_is_committable(self, tmp_path):
        """`*.csv` in .gitignore would otherwise swallow deliverable 4a itself."""
        target = REPO_ROOT / "runs" / "examples" / "_pytest_probe" / "final" / "submission.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("row_id,user_id,video_id,score\n0,1,2,0.5\n")
        try:
            assert not ignored(target), (
                "runs/examples/**/*.csv is ignored again — final/submission.csv IS "
                "deliverable 4a, and it would be missing from the public repo"
            )
        finally:
            for p in (target, target.parent, target.parent.parent):
                if p.exists():
                    p.unlink() if p.is_file() else p.rmdir()

    @pytest.mark.parametrize("name", ["journal.jsonl", "interventions.md", "summary.json"])
    def test_the_log_deliverables_are_committable(self, name):
        """Items 3, 3b and 4c."""
        target = REPO_ROOT / "runs" / "examples" / "_pytest_probe" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
        try:
            assert not ignored(target)
        finally:
            target.unlink()
            target.parent.rmdir()

    def test_an_ordinary_run_is_still_ignored(self):
        """The exemption must not accidentally start committing every working run."""
        assert ignored(REPO_ROOT / "runs" / "r20260831-9999" / "journal.jsonl")


class TestArchiveRun:
    def test_it_keeps_the_graded_artifacts(self, tmp_path):
        dest = archive(make_run(tmp_path), dest_root=tmp_path / "examples")
        assert (dest / "journal.jsonl").is_file()          # item 3
        assert (dest / "interventions.md").is_file()       # item 3b
        assert (dest / "final" / "submission.csv").is_file()   # item 4a
        assert (dest / "summary.json").is_file()           # item 4c
        assert (dest / "nodes" / "n002" / "pipeline.py").is_file(), (
            "the per-node pipelines are the evidence of what the agent chose to write"
        )

    def test_it_drops_the_bulk_that_would_make_the_commit_huge(self, tmp_path):
        dest = archive(make_run(tmp_path), dest_root=tmp_path / "examples")
        assert not (dest / "nodes" / "n000" / "submission.csv").exists(), (
            "per-node submissions are ~170k rows each; a full run would be hundreds of MB"
        )
        assert not (dest / "nodes" / "n000" / "stdout.log").exists()

    def test_it_never_writes_back_to_the_source_run(self, tmp_path):
        run = make_run(tmp_path)
        before = {p.relative_to(run) for p in run.rglob("*")}
        archive(run, dest_root=tmp_path / "examples")
        assert {p.relative_to(run) for p in run.rglob("*")} == before

    def test_it_refuses_to_clobber_an_existing_archive(self, tmp_path):
        run = make_run(tmp_path)
        archive(run, dest_root=tmp_path / "examples")
        with pytest.raises(SystemExit, match="--force"):
            archive(run, dest_root=tmp_path / "examples")
        archive(run, dest_root=tmp_path / "examples", force=True)  # explicit is fine

    def test_it_rejects_a_directory_that_is_not_a_run(self, tmp_path):
        (tmp_path / "not_a_run").mkdir()
        with pytest.raises(SystemExit, match="journal.jsonl"):
            archive(tmp_path / "not_a_run", dest_root=tmp_path / "examples")

    def test_a_crashed_run_with_no_summary_still_archives(self, tmp_path):
        """r20260831-0741 crashed at finalisation. Its journal is still deliverable 3."""
        run = make_run(tmp_path)
        (run / "summary.json").unlink()
        (run / "final" / "submission.csv").unlink()
        dest = archive(run, dest_root=tmp_path / "examples")
        assert (dest / "journal.jsonl").is_file()
