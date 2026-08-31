"""Rank by how often the item appears, instead of by nothing at all.

Exposure in a feed is not uniform: a small set of videos accounts for a large share of
impressions, and impression count is a crude proxy for the ranker's own prior that the
item is worth showing. Ordering a user's impressions by item frequency should therefore
beat the constant control, without using a single label.

For a task with no groups the same instinct is a conditional mean: predict the training
mean of the target within the row's own category rather than the mean over everything.

Served offline by `ReplayAgent`; runs under the frozen pipeline CLI like any other.
"""

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict

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
#: Widest cardinality still worth treating as a category rather than as free text.
MAX_LEVELS = 60


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
        return float(value)
    except (TypeError, ValueError):
        return None


def pick_category(rows, target, exclude):
    """A column that partitions the rows into a usable number of groups.

    Anything with one level says nothing, and anything near one level per row memorises
    the training set instead of describing it — so both ends are refused.
    """
    if not rows:
        return None
    for column in rows[0]:
        if column == target or column in exclude:
            continue
        levels = {r.get(column) for r in rows}
        if 1 < len(levels) <= MAX_LEVELS:
            return column
    return None


def group_means(data_dir, target):
    """(column, {level: mean}, global mean) learned from the training split only."""
    train = table_path(data_dir, "train")
    rows = read_table(train) if train else []
    values = [(r, numeric(r.get(target))) for r in rows]
    values = [(r, v) for r, v in values if v is not None]
    if not values:
        return None, {}, 0.0
    overall = sum(v for _, v in values) / len(values)
    column = pick_category([r for r, _ in values], target, set(ID_COLUMNS))
    if column is None:
        return None, {}, overall
    totals = defaultdict(lambda: [0.0, 0])
    for record, value in values:
        slot = totals[record.get(column)]
        slot[0] += value
        slot[1] += 1
    return column, {k: t / n for k, (t, n) in totals.items()}, overall


def main():
    t0 = time.time()
    args = parse_args()
    split = {"val": "valid"}.get(args.split, args.split)
    layout, ids, records = load_eval(args.data_dir, split)

    if layout == "starter_kit":
        item_at = len(ID_COLUMNS) - 1
        counts = Counter(row[item_at] for row in ids)
        preds = [float(counts[row[item_at]]) for row in ids]
        notes = f"item popularity over {len(counts)} distinct items"
    else:
        target = detect_target(args.data_dir)
        column, means, overall = group_means(args.data_dir, target) if target else (None, {}, 0.0)
        preds = [
            means.get(r.get(column), overall) if column else overall for r in (records or [])
        ]
        notes = (
            f"mean {target} within {column} ({len(means)} groups)"
            if column
            else f"global mean {target}"
        )

    write_submission(args.out_dir, ids, preds)
    emit(t0, len(ids), f"{notes}, layout={layout}, split={split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
