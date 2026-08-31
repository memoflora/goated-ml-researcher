"""Use the training labels, not just the exposure counts, and shrink the noisy ones.

Impression count says an item was shown often; it does not say anyone watched it. The
label is available on the training split, so the per-item long-view rate is a strictly
better prior than frequency — except in the tail, where an item with three impressions and
two long views would otherwise outrank everything on 0.67. Shrinking each rate towards the
global rate in proportion to how little evidence supports it is what makes the tail safe:
a well-observed item keeps its own rate, a barely-observed one keeps the prior.

For a task with no groups the same step is to stop predicting a mean and start predicting
a function: an ordinary least-squares fit over the numeric columns, which is the smallest
model that can use more than one feature at a time.

Served offline by `ReplayAgent`; runs under the frozen pipeline CLI like any other.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np

# `ReplayAgent` substitutes the task's real submission header here. The literal below is
# left in place when the file is run directly, so it also works standalone.
SUBMISSION_HEADER = "__SUBMISSION_HEADER__"
COLUMNS = (
    SUBMISSION_HEADER if "," in SUBMISSION_HEADER else "row_id,user_id,video_id,score"
).split(",")
ID_COLUMNS = COLUMNS[1:-1]

LOGS = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
RANGES = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
LABEL = "long_view"
#: Pseudo-counts of the global rate mixed into every item. Larger = more shrinkage.
PRIOR_STRENGTH = 20.0
#: Ridge term, so a constant or duplicated column cannot make the normal equations blow up.
RIDGE = 1e-6


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample", type=float, default=None)
    return ap.parse_args()


def is_starter_kit(data_dir):
    return all(os.path.isfile(os.path.join(data_dir, n)) for n in LOGS)


def iter_logs(data_dir, split, columns):
    lo, hi = RANGES[split]
    for name in LOGS:
        with open(os.path.join(data_dir, name), newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            head = next(reader)
            date_at = head.index("date")
            at = [head.index(c) for c in columns]
            for rec in reader:
                if lo <= int(rec[date_at]) <= hi:
                    yield [rec[i] for i in at]


def table_path(data_dir, split):
    names = [f"{split}.csv"] + (["val.csv"] if split == "valid" else [])
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    return None


def read_table(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def header_of(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh), []) or []


def detect_target(data_dir):
    train, test = table_path(data_dir, "train"), table_path(data_dir, "test")
    if not train or not test:
        return None
    held_out = set(header_of(test))
    extra = [c for c in header_of(train) if c not in held_out]
    return extra[-1] if extra else None


def load_eval(data_dir, split):
    if is_starter_kit(data_dir):
        return "starter_kit", list(iter_logs(data_dir, split, ID_COLUMNS)), None
    path = table_path(data_dir, split)
    if path is None:
        raise FileNotFoundError(f"no {split} split under {data_dir}")
    records = read_table(path)
    return "table", [[str(r[c]) for c in ID_COLUMNS] for r in records], records


def write_submission(out_dir, ids, preds):
    os.makedirs(out_dir, exist_ok=True)
    with open(
        os.path.join(out_dir, "submission.csv"), "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row_id, (row, pred) in enumerate(zip(ids, preds, strict=True)):
            writer.writerow([row_id, *row, f"{pred:.6f}"])


def emit(t0, n_rows, notes):
    print(
        "RESULT_JSON "
        + json.dumps(
            {"n_rows": n_rows, "train_seconds": round(time.time() - t0, 3), "notes": notes}
        )
    )


def numeric(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


# --------------------------------------------------------------------------- #
# ranking: shrunken per-item rate, learned on train only
# --------------------------------------------------------------------------- #

def item_rates(data_dir):
    item_column = ID_COLUMNS[-1]
    positives, totals = defaultdict(float), defaultdict(int)
    seen, hits = 0, 0.0
    for item, label in iter_logs(data_dir, "train", [item_column, LABEL]):
        value = 1.0 if label not in ("0", "", "0.0") else 0.0
        positives[item] += value
        totals[item] += 1
        seen += 1
        hits += value
    prior = hits / seen if seen else 0.0
    rates = {
        item: (positives[item] + PRIOR_STRENGTH * prior) / (totals[item] + PRIOR_STRENGTH)
        for item in totals
    }
    return rates, prior


# --------------------------------------------------------------------------- #
# regression: least squares over whatever columns are numeric in both splits
# --------------------------------------------------------------------------- #

def numeric_columns(rows, target):
    """Columns parseable as numbers on most rows. A column that is numeric only
    sometimes is a categorical wearing a disguise, and it is left out."""
    if not rows:
        return []
    out = []
    for column in rows[0]:
        if column == target or column in ID_COLUMNS:
            continue
        parsed = [numeric(r.get(column)) for r in rows]
        if sum(v is not None for v in parsed) >= 0.9 * len(rows):
            out.append(column)
    return out


def design(rows, columns, fill):
    matrix = np.ones((len(rows), len(columns) + 1), dtype=np.float64)
    for j, column in enumerate(columns):
        for i, record in enumerate(rows):
            value = numeric(record.get(column))
            matrix[i, j + 1] = fill[column] if value is None else value
    return matrix


def least_squares(data_dir, target, eval_rows):
    train = table_path(data_dir, "train")
    rows = [r for r in (read_table(train) if train else []) if numeric(r.get(target)) is not None]
    if not rows:
        return None, "no usable training rows"
    columns = numeric_columns(rows, target)
    y = np.array([numeric(r[target]) for r in rows], dtype=np.float64)
    if not columns:
        return np.full(len(eval_rows), float(y.mean())), "no numeric columns; global mean"

    fill = {}
    for column in columns:
        seen = [v for v in (numeric(r.get(column)) for r in rows) if v is not None]
        fill[column] = float(np.mean(seen)) if seen else 0.0

    x = design(rows, columns, fill)
    gram = x.T @ x + RIDGE * np.eye(x.shape[1])
    beta = np.linalg.solve(gram, x.T @ y)
    preds = design(eval_rows, columns, fill) @ beta
    return preds, f"least squares on {len(columns)} numeric column(s)"


def main():
    t0 = time.time()
    args = parse_args()
    split = {"val": "valid"}.get(args.split, args.split)
    layout, ids, records = load_eval(args.data_dir, split)

    if layout == "starter_kit":
        rates, prior = item_rates(args.data_dir)
        item_at = len(ID_COLUMNS) - 1
        preds = [rates.get(row[item_at], prior) for row in ids]
        notes = (
            f"shrunken per-item rate over {len(rates)} items "
            f"(prior {prior:.4f}, strength {PRIOR_STRENGTH:g})"
        )
    else:
        target = detect_target(args.data_dir)
        if not target:
            raise ValueError(f"could not identify the target column under {args.data_dir}")
        preds, notes = least_squares(args.data_dir, target, records or [])
        if preds is None:
            raise ValueError(notes)
        preds = [float(v) for v in preds]

    write_submission(args.out_dir, ids, preds)
    emit(t0, len(ids), f"{notes}, layout={layout}, split={split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
