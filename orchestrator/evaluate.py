"""Scoring and submission validation — the pinned metric seam.

Two functions, per `contracts.md` §3:

    score(submission: Path, split: str) -> dict[str, float]
    validate(submission: Path, split: str) -> tuple[bool, str]

**The metrics are never reimplemented here.** `score` delegates to the vendored
`vendor/starter_kit/evaluate.py`, which is the sole authority on the conventions
(zero-positive users count as nDCG 0 and are included in the mean; GAUC covers only users
with `0 < positives < impressions`, weighted by positive count; gain = `2^rel - 1`).
Reimplementing any of that would make our whole validation signal a lie.

`validate` mirrors the checks in `vendor/starter_kit/submit.py::read_submission` one for
one, but returns `(ok, message)` instead of raising, because the orchestrator turns a
failed check into `error_class="contract"` and routes it to the repair loop rather than
crashing the run. `tests/test_evaluator.py` asserts parity with `submit.py` on every
rejection case.

Hidden-test guard
-----------------
KuaiRand-Pure ships the test labels, so `score(..., "test")` *would* work locally. Using
it during development is the one thing that invalidates our own stopping signal, so it is
refused unless `ALLOW_TEST_SCORING=1` is set explicitly. The final submission is built by
rerunning the validation-best node with `--split test` and checking it with `validate`,
which needs no labels at all.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

from orchestrator.splits import STARTER_KIT, Split, get_split, normalise_split

HEADER = ["row_id", "user_id", "video_id", "score"]

#: Journal/metric key names. Lowercase, matching the journal schema in contracts.md §5.
METRIC_KEYS = ("gauc", "ndcg@5", "primary")

TEST_SCORING_ENV = "ALLOW_TEST_SCORING"


class SubmissionError(ValueError):
    """A submission that fails format or alignment checks."""


def _starter_kit_evaluate():
    """Import the vendored `evaluate.evaluate` without leaking a very generic module name."""
    sys.path.insert(0, str(STARTER_KIT))
    try:
        import evaluate as starter_evaluate  # type: ignore[import-not-found]

        return starter_evaluate.evaluate
    finally:
        if sys.path and sys.path[0] == str(STARTER_KIT):
            sys.path.pop(0)
        sys.modules.pop("evaluate", None)


def _read_scores(submission: Path, split_obj: Split) -> np.ndarray:
    """Parse and fully validate a submission, returning scores in `row_id` order.

    Mirrors `submit.py::read_submission`, including the order the checks fire in, so the
    first error we report is the same one the official checker would report.
    """
    users, videos = split_obj.user_ids, split_obj.video_ids
    n_expected = len(split_obj)

    with open(submission, newline="") as fh:
        reader = csv.reader(fh)
        head = next(reader, None)
        if head != HEADER:
            raise SubmissionError(f"header must be {','.join(HEADER)}, got {head}")

        scores = np.empty(n_expected, dtype=np.float64)
        n = 0
        for ln, rec in enumerate(reader, start=2):
            if len(rec) != 4:
                raise SubmissionError(f"line {ln} has {len(rec)} fields, expected 4")
            rid, uid, vid, sc = rec
            if n >= n_expected:
                raise SubmissionError(
                    f"line {ln}: more rows than the {split_obj.name} split ({n_expected} rows)"
                )
            try:
                rid_i = int(rid)
            except ValueError:
                raise SubmissionError(f"line {ln}: row_id {rid!r} is not an integer") from None
            if rid_i != n:
                raise SubmissionError(
                    f"line {ln}: row_id={rid}, expected {n} (must increase from 0 with no gaps)"
                )
            try:
                uid_i, vid_i = int(uid), int(vid)
            except ValueError:
                raise SubmissionError(
                    f"line {ln}: user_id/video_id {uid!r}/{vid!r} are not integers"
                ) from None
            if uid_i != users[n] or vid_i != videos[n]:
                raise SubmissionError(
                    f"line {ln} misaligned: submission has ({uid},{vid}), "
                    f"{split_obj.name} row {n} is ({users[n]},{videos[n]})"
                )
            try:
                v = float(sc)
            except ValueError:
                raise SubmissionError(f"line {ln}: score {sc!r} is not a number") from None
            if math.isnan(v) or math.isinf(v):
                raise SubmissionError(f"line {ln}: score is NaN/Inf, which is not allowed")
            scores[n] = v
            n += 1

    if n != n_expected:
        raise SubmissionError(
            f"submission has {n} rows, {split_obj.name} split has {n_expected}"
        )
    return scores


def validate(submission: Path | str, split: str) -> tuple[bool, str]:
    """Check format and alignment. Returns `(ok, message)`; never raises on a bad file.

    Rejects: wrong header, wrong field count, row_id gaps or non-integers, row-count
    mismatch in either direction, user/video misalignment, non-numeric or NaN/Inf scores.
    """
    submission = Path(submission)
    try:
        split_obj = get_split(split)
    except (ValueError, FileNotFoundError) as exc:
        return False, str(exc)
    if not submission.is_file():
        return False, f"submission not found: {submission}"
    try:
        _read_scores(submission, split_obj)
    except SubmissionError as exc:
        return False, str(exc)
    except OSError as exc:
        return False, f"could not read submission: {exc}"
    return True, f"ok: {len(split_obj):,d} rows, split={split_obj.name}"


def score(submission: Path | str, split: str) -> dict[str, float]:
    """Validate then score a submission. Returns `{'gauc','ndcg@5','primary'}`.

    Raises `SubmissionError` if the file is malformed — the orchestrator treats that as
    `error_class="contract"` and sends the node to the repair loop.
    """
    submission = Path(submission)
    name = normalise_split(split)
    if name == "test" and os.environ.get(TEST_SCORING_ENV) != "1":
        raise SubmissionError(
            "refusing to score the hidden test split during development. "
            "Development uses train + validation only; scoring test here would corrupt "
            f"our stopping signal. Set {TEST_SCORING_ENV}=1 only for a post-hoc audit."
        )
    split_obj = get_split(name)
    scores = _read_scores(submission, split_obj)
    return score_arrays(split_obj.user_ids, split_obj.labels, scores)


def score_arrays(
    user_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Score raw arrays with the vendored evaluator. Used by `score` and by calibration."""
    raw = _starter_kit_evaluate()(
        [int(u) for u in user_ids],
        [int(y) for y in labels],
        [float(s) for s in scores],
    )
    return {
        "gauc": float(raw["GAUC"]),
        "ndcg@5": float(raw["nDCG@5"]),
        "primary": float(raw["primary"]),
    }


def delta_vs_baseline(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """Absolute per-metric improvement over a baseline — the journal's `delta_vs_baseline`."""
    return {k: float(metrics[k] - baseline[k]) for k in METRIC_KEYS if k in metrics}
