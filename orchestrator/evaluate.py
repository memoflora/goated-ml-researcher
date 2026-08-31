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

Every rejection message is written for that repair loop, not for a human reading a
terminal: the LLM is handed the text and nothing else — never the file — so each message
carries the diagnosis *and* the rule it broke. `submit.py` says "row_id=99, expected 3";
we add what row_id means and which operation (a sort, a dedupe, a merge) produces that
symptom. The KuaiRand path and the generic task path share one set of message builders so
that an agent which learned to fix one task's contract error can fix any task's.

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


# --------------------------------------------------------------------------- messages
# Every rejection below is read by an LLM that has to repair the pipeline from the text
# alone: it never sees the submission file. So each message says three things — what was
# found, what was expected, and the shape of the fix. `tests/test_evaluator.py::
# TestRepairableMessages` pins that third part, because it is the one an ordinary
# "raise ValueError(...)" leaves out.


def _header_message(head: list[str] | None, expected: list[str]) -> str:
    want = ",".join(expected)
    if head is None:
        return (
            f"submission is empty: the first line must be the header {want}, followed by "
            "one row per evaluation-split row"
        )
    parts = [f"header must be {want}, got {','.join(head)}"]
    if len(head) != len(expected):
        parts.append(f"{len(head)} columns, expected {len(expected)}")
    for i, (got_col, want_col) in enumerate(zip(head, expected, strict=False), start=1):
        if got_col != want_col:
            parts.append(f"column {i} is {got_col!r}, expected {want_col!r}")
            break
    missing = [c for c in expected if c not in head]
    if missing:
        parts.append("missing " + ", ".join(repr(c) for c in missing))
    parts.append("write the header exactly, in this order, once, as the first line")
    return "; ".join(parts)


def _field_count_message(ln: int, got: int, expected: list[str]) -> str:
    return (
        f"line {ln} has {got} fields, expected {len(expected)} "
        f"({','.join(expected)}) — an unquoted comma inside a value, a missing value, or a "
        "stray trailing comma will do this"
    )


def _row_id_message(ln: int, rid: str | int, n: int, split_name: str) -> str:
    return (
        f"line {ln}: row_id={rid}, expected {n} (must increase from 0 with no gaps). "
        f"row_id is the 0-based position of the row in the {split_name} split: start at 0 "
        "and add exactly 1 per line. Never sort, shuffle, deduplicate or reindex the "
        "evaluation rows before writing them"
    )


def _too_many_rows_message(ln: int, split_name: str, n_expected: int) -> str:
    return (
        f"line {ln}: more rows than the {split_name} split, which has exactly "
        f"{n_expected:,d} rows. Write one row per evaluation-split row and stop — do not "
        "append extra rows, and do not write the header twice"
    )


def _row_count_message(n: int, split_name: str, n_expected: int) -> str:
    return (
        f"submission has {n:,d} rows, {split_name} split has {n_expected:,d} "
        f"({n_expected - n:,d} missing). Write exactly one row per evaluation-split row, "
        "in the split's own order — a filtered, deduplicated or subsampled evaluation set "
        "is the usual cause (--subsample must only shrink training data)"
    )


def _not_a_number_message(ln: int, column: str, raw: str) -> str:
    return (
        f"line {ln}: {column} {raw!r} is not a number — it must be a finite decimal "
        "number, not a label, a blank, or a formatted string"
    )


def _non_finite_message(ln: int, column: str, raw: str) -> str:
    return (
        f"line {ln}: {column} is NaN/Inf, which is not allowed (value {raw!r}). Every "
        "prediction must be finite — check for division by zero, log/exp overflow, an "
        "unfilled default, or a NaN that propagated out of a join or a groupby"
    )


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

    with open(submission, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        head = next(reader, None)
        if head != HEADER:
            raise SubmissionError(_header_message(head, HEADER))

        scores = np.empty(n_expected, dtype=np.float64)
        n = 0
        for ln, rec in enumerate(reader, start=2):
            if len(rec) != 4:
                raise SubmissionError(_field_count_message(ln, len(rec), HEADER))
            rid, uid, vid, sc = rec
            if n >= n_expected:
                raise SubmissionError(
                    _too_many_rows_message(ln, split_obj.name, n_expected)
                )
            try:
                rid_i = int(rid)
            except ValueError:
                raise SubmissionError(
                    f"line {ln}: row_id {rid!r} is not an integer — it must be the 0-based "
                    f"position of the row in the {split_obj.name} split ({n} on this line)"
                ) from None
            if rid_i != n:
                raise SubmissionError(_row_id_message(ln, rid, n, split_obj.name))
            try:
                uid_i, vid_i = int(uid), int(vid)
            except ValueError:
                raise SubmissionError(
                    f"line {ln}: user_id/video_id {uid!r}/{vid!r} are not integers — copy "
                    f"them through from the {split_obj.name} rows unchanged"
                ) from None
            if uid_i != users[n] or vid_i != videos[n]:
                raise SubmissionError(
                    f"line {ln} misaligned: submission has ({uid},{vid}), "
                    f"{split_obj.name} row {n} is ({users[n]},{videos[n]}). Emit the "
                    "evaluation rows in the split's own order; (user_id, video_id) is not "
                    "unique, so never merge or key on it"
                )
            try:
                v = float(sc)
            except ValueError:
                raise SubmissionError(_not_a_number_message(ln, "score", sc)) from None
            if math.isnan(v) or math.isinf(v):
                raise SubmissionError(_non_finite_message(ln, "score", sc))
            scores[n] = v
            n += 1

    if n != n_expected:
        raise SubmissionError(_row_count_message(n, split_obj.name, n_expected))
    return scores


def _is_starter_kit(task) -> bool:
    """KuaiRand keeps its own path: the organisers' loader and the organisers' metrics."""
    return task is None or getattr(task.data, "loader", "") == "starter_kit"


def _resolve(task):
    """Accept a TaskConfig, a task name, or None (meaning KuaiRand-Pure)."""
    if task is None or not isinstance(task, str):
        return task
    from orchestrator.taskspec import load_task

    return load_task(task)


def _read_predictions(submission: Path, task, arrays) -> np.ndarray:
    """Generic counterpart of `_read_scores`, driven by the task's submission schema.

    Same checks in the same order — header, field count, row_id contiguity, id alignment,
    finite numbers — so an agent that fixes one task's contract error has learned how to
    fix every task's.
    """
    header = list(task.submission_columns)
    pred_at = header.index(task.prediction_column)
    id_cols = [(i, c) for i, c in enumerate(header) if c in arrays.ids]
    n_expected = len(arrays)

    with open(submission, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        head = next(reader, None)
        if head != header:
            raise SubmissionError(_header_message(head, header))

        preds = np.empty(n_expected, dtype=np.float64)
        n = 0
        for ln, rec in enumerate(reader, start=2):
            if len(rec) != len(header):
                raise SubmissionError(_field_count_message(ln, len(rec), header))
            if n >= n_expected:
                raise SubmissionError(_too_many_rows_message(ln, arrays.name, n_expected))
            try:
                rid = int(rec[0])
            except ValueError:
                raise SubmissionError(
                    f"line {ln}: row_id {rec[0]!r} is not an integer — it must be the "
                    f"0-based position of the row in the {arrays.name} split "
                    f"({n} on this line)"
                ) from None
            if rid != n:
                raise SubmissionError(_row_id_message(ln, rid, n, arrays.name))
            for i, col in id_cols:
                want = arrays.ids[col][n]
                if str(rec[i]) != str(want):
                    raise SubmissionError(
                        f"line {ln} misaligned: {col}={rec[i]!r}, "
                        f"{arrays.name} row {n} has {want!r}. Emit the evaluation rows in "
                        "the split's own order; the id columns are copied through for "
                        "checking, not used as a key"
                    )
            try:
                v = float(rec[pred_at])
            except ValueError:
                raise SubmissionError(
                    _not_a_number_message(ln, task.prediction_column, rec[pred_at])
                ) from None
            if math.isnan(v) or math.isinf(v):
                raise SubmissionError(
                    _non_finite_message(ln, task.prediction_column, rec[pred_at])
                )
            preds[n] = v
            n += 1

    if n != n_expected:
        raise SubmissionError(_row_count_message(n, arrays.name, n_expected))
    return preds


def validate(submission: Path | str, split: str, task=None) -> tuple[bool, str]:
    """Check format and alignment. Returns `(ok, message)`; never raises on a bad file.

    Rejects: wrong header, wrong field count, row_id gaps or non-integers, row-count
    mismatch in either direction, id misalignment, non-numeric or NaN/Inf predictions.
    """
    submission = Path(submission)
    task = _resolve(task)
    if not submission.is_file():
        return False, (
            f"submission not found: {submission} — the pipeline must write "
            "submission.csv into the directory given by --out-dir before it exits 0"
        )

    if _is_starter_kit(task):
        try:
            split_obj = get_split(split)
        except (ValueError, FileNotFoundError) as exc:
            return False, str(exc)
        try:
            _read_scores(submission, split_obj)
        except SubmissionError as exc:
            return False, str(exc)
        except OSError as exc:
            return False, f"could not read submission: {exc}"
        return True, f"ok: {len(split_obj):,d} rows, split={split_obj.name}"

    from orchestrator import datasource as ds

    try:
        arrays = ds.eval_arrays(task, split)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        return False, str(exc)
    try:
        _read_predictions(submission, task, arrays)
    except SubmissionError as exc:
        return False, str(exc)
    except OSError as exc:
        return False, f"could not read submission: {exc}"
    return True, f"ok: {len(arrays):,d} rows, split={arrays.name}"


def score(submission: Path | str, split: str, task=None) -> dict[str, float]:
    """Validate then score a submission, returning every metric the task reports plus
    `primary` (always oriented so that higher is better).

    Raises `SubmissionError` if the file is malformed — the orchestrator treats that as
    `error_class="contract"` and sends the node to the repair loop.
    """
    submission = Path(submission)
    task = _resolve(task)
    name = normalise_split(split)
    if name == "test" and os.environ.get(TEST_SCORING_ENV) != "1":
        raise SubmissionError(
            "refusing to score the held-out test split during development. "
            "Development uses train + validation only; scoring test here would corrupt "
            f"our stopping signal. Set {TEST_SCORING_ENV}=1 only for a post-hoc audit."
        )

    if _is_starter_kit(task):
        split_obj = get_split(name)
        scores = _read_scores(submission, split_obj)
        return score_arrays(split_obj.user_ids, split_obj.labels, scores)

    from orchestrator import datasource as ds
    from orchestrator import metrics as M

    arrays = ds.eval_arrays(task, name)
    preds = _read_predictions(submission, task, arrays)
    if not np.isfinite(arrays.target).all():
        raise SubmissionError(
            f"the {name} split has no usable labels, so it cannot be scored here"
        )
    out: dict[str, float] = {}
    for metric_name in task.all_metrics:
        out[metric_name] = M.compute(metric_name, arrays.target, preds, arrays.groups)
    out["primary"] = M.primary_of(out, task.primary_parts)
    return out


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


def delta_vs_baseline(
    metrics: dict[str, float], baseline: dict[str, float], keys: tuple[str, ...] | None = None
) -> dict[str, float]:
    """Absolute per-metric improvement over a baseline — the journal's `delta_vs_baseline`.

    Signed as the raw difference, so for a lower-is-better metric a *negative* delta is an
    improvement. `primary` is always oriented the other way, which is why the report shows
    both rather than trying to collapse them.
    """
    if keys is None:
        keys = tuple(k for k in metrics if k in baseline) or METRIC_KEYS
    return {k: float(metrics[k] - baseline[k]) for k in keys if k in metrics and k in baseline}
