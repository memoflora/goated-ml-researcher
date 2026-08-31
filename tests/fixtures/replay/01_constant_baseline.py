"""Establish the floor before anything clever: predict one constant for every row.

A constant is worthless as a model and invaluable as a control. It scores exactly what a
pipeline scores when it has learned nothing, so every later number has something honest to
be compared against — and, more usefully on iteration one, it proves the whole path works:
the split is read in the right order, the submission aligns, and the evaluator accepts it.
Debugging a clever model and a broken contract at the same time is how a run dies.

Served offline by `ReplayAgent`; runs under the frozen pipeline CLI like any other.
"""

import argparse
import csv
import json
import os
import time

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
    """Rows of a KuaiRand split, in `data.load()` order: file order, then the date filter.

    That order is the definition of `row_id`, so it is reproduced rather than approximated.
    """
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
    """The target is the column train.csv carries and test.csv does not, by construction."""
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


def train_target_mean(data_dir):
    target, train = detect_target(data_dir), table_path(data_dir, "train")
    if not target or not train:
        return None
    total, n = 0.0, 0
    for rec in read_table(train):
        try:
            total += float(rec[target])
        except (TypeError, ValueError):
            continue
        n += 1
    return total / n if n else None


def main():
    t0 = time.time()
    args = parse_args()
    split = {"val": "valid"}.get(args.split, args.split)
    layout, ids, _records = load_eval(args.data_dir, split)

    if layout == "starter_kit":
        # Every row gets the same score, so no user's ordering is changed: this is the
        # 0.5-GAUC control, on purpose.
        value, what = 0.0, "constant 0.0"
    else:
        mean = train_target_mean(args.data_dir)
        value = 0.0 if mean is None else mean
        what = f"constant train-set mean {value:.4f}"

    write_submission(args.out_dir, ids, [value] * len(ids))
    emit(t0, len(ids), f"constant baseline ({what}), layout={layout}, split={split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
