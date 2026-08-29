"""Shared test scaffolding. B owns the sandbox/agent/fault tests that use it."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contracts import Node  # noqa: E402

FIXTURE_PIPELINES = Path(__file__).parent / "fixtures" / "pipelines"


@pytest.fixture
def make_node(tmp_path):
    """Materialise a node workspace containing one of the fault fixtures."""
    counter = {"n": 0}

    def _make(fixture: str, *, kind: str = "draft") -> Node:
        idx = counter["n"]
        counter["n"] += 1
        ws = tmp_path / "nodes" / f"n{idx:03d}"
        ws.mkdir(parents=True)
        shutil.copy(FIXTURE_PIPELINES / fixture, ws / "pipeline.py")
        shutil.copy(FIXTURE_PIPELINES / "_cli.py", ws / "_cli.py")
        return Node(id=f"n{idx:03d}", parent_id=None, kind=kind,
                    iteration=idx, workspace=ws)

    return _make


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d
