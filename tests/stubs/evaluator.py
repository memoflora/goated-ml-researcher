"""StubEvaluator — stands in for C's `evaluate.py` until it lands.

    score(submission, split) -> {"gauc": ..., "ndcg@5": ..., "primary": ...}
    validate(submission, split) -> (ok, message)

`score` reads the `stub_metrics.json` sidecar the StubExecutor wrote next to the
submission, so the numbers a test asserts on are the numbers it scripted.
`validate` is a genuine (if minimal) schema check: header, row_id gaps, finite
scores. C's real one is stricter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HEADER = "row_id,user_id,video_id,score"


class StubEvaluator:
    def score(self, submission: Path | str, split: str = "val") -> dict[str, float]:
        sidecar = Path(submission).parent / "stub_metrics.json"
        if sidecar.exists():
            return json.loads(sidecar.read_text(encoding="utf-8"))
        return {"gauc": 0.5, "ndcg@5": 0.5, "primary": 0.5}

    def validate(self, submission: Path | str, split: str = "val") -> tuple[bool, str]:
        p = Path(submission)
        if not p.exists():
            return False, f"missing submission: {p}"
        with p.open("r", encoding="utf-8") as fh:
            header = fh.readline().strip()
            if header != HEADER:
                return False, f"bad header {header!r}, expected {HEADER!r}"
            for expected, line in enumerate(fh):
                parts = line.strip().split(",")
                if len(parts) != 4:
                    return False, f"row {expected}: expected 4 fields, got {len(parts)}"
                if int(parts[0]) != expected:
                    return False, f"row_id gap at {expected}: got {parts[0]}"
                value = float(parts[3])
                if math.isnan(value) or math.isinf(value):
                    return False, f"row {expected}: non-finite score {parts[3]}"
        return True, "ok"
