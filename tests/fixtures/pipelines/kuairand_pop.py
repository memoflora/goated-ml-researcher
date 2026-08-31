"""Item popularity on KuaiRand-Pure — a REAL reference pipeline, not a fault fixture.

This is the organisers' `baseline.py --model pop` re-expressed against the frozen pipeline
contract (`references/contracts.md` §1):

    python pipeline.py --data-dir DIR --out-dir DIR --split {val,test} --seed N [--subsample F]

It exists to prove the whole seam end to end on real data:

    sandbox.run  ->  submission.csv  ->  evaluate.validate  ->  evaluate.score

`tests/test_evaluator.py::TestRealPipelines` runs it and asserts the published validation
numbers (GAUC 0.6387 / nDCG@5 0.5227 / primary 0.5807). If that assertion ever drifts, the
harness is wrong and every metric the orchestrator reports is wrong with it.

Self-contained by contract: no imports from `orchestrator/` or `vendor/`, only libraries
from `requirements-pipeline.txt`. The row ordering, the date split and the label rule are
therefore re-derived here rather than imported — they are duplicated *deliberately*, and
`test_pop_row_order_matches_the_starter_kit_loader` pins the duplicate against the
authority so the two can never silently disagree.

Scoring never happens in here. The pipeline emits scores; the orchestrator grades them.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

# --- facts about the dataset, from the organisers' data.py -------------------------------
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",   # read first ...
    "log_standard_4_22_to_5_08_pure.csv",   # ... then this one. Order is the row_id order.
)
LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
# The starter kit calls the middle split "valid"; the CLI contract says "val".
SPLIT_ALIASES = {"val": "valid", "valid": "valid", "test": "test", "train": "train"}

PRIOR = 20.0  # smoothing strength, same value the official baseline uses


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="KuaiRand-Pure item-popularity baseline")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", required=True, choices=["val", "valid", "test"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--subsample", type=float, default=None)
    return p.parse_args(argv)


def resolve_data_dir(data_dir: str) -> str:
    """Accept either the leaf `.../KuaiRand-Pure/data` or a parent that contains it."""
    candidates = (
        data_dir,
        os.path.join(data_dir, "KuaiRand-Pure", "data"),
        os.path.join(data_dir, "data"),
    )
    for c in candidates:
        if os.path.isfile(os.path.join(c, LOG_FILES[0])):
            return c
    raise FileNotFoundError(
        "could not find %s under %r (tried: %s)"
        % (LOG_FILES[0], data_dir, ", ".join(repr(c) for c in candidates))
    )


def load_logs(data_dir: str) -> pd.DataFrame:
    """The two standard logs, concatenated in file order, with the label already binarised.

    Only the four columns popularity needs are parsed. Row order is the file order — that
    is what `row_id` indexes into, so nothing here may sort, dedupe or reindex.
    """
    frames = []
    for fname in LOG_FILES:
        frames.append(
            pd.read_csv(
                os.path.join(data_dir, fname),
                usecols=["user_id", "video_id", "date", LABEL],
                dtype={"user_id": np.int64, "video_id": np.int64,
                       "date": np.int64, LABEL: np.int64},
                encoding="utf-8",
            )
        )
    df = pd.concat(frames, ignore_index=True)
    # data.py: `1 if r[LABEL] != '0' else 0` — any non-zero counts as a long view.
    df["y"] = (df[LABEL].to_numpy() != 0).astype(np.int8)
    return df


def slice_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    lo, hi = SPLITS[name]
    date = df["date"].to_numpy()
    return df.loc[(date >= lo) & (date <= hi)]


def subsample_users(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Keep a random fraction of whole users.

    Whole groups, never individual rows: popularity is a per-item statistic, but the metric
    is computed *within a user*, so a fixture that row-sampled here would model a different
    training distribution from the one the contract describes.
    """
    if frac is None or frac >= 1.0:
        return df
    users = np.unique(df["user_id"].to_numpy())
    rng = np.random.default_rng(seed)
    keep_n = max(1, int(round(len(users) * float(frac))))
    keep = set(rng.choice(users, size=keep_n, replace=False).tolist())
    mask = np.fromiter((u in keep for u in df["user_id"].to_numpy()), dtype=bool,
                       count=len(df))
    return df.loc[mask]


def fit_popularity(train: pd.DataFrame) -> tuple[dict[int, float], float]:
    """Smoothed per-video positive rate: (pos + prior*gmean) / (imp + prior)."""
    vids = train["video_id"].to_numpy()
    ys = train["y"].to_numpy()
    order = np.argsort(vids, kind="stable")
    vs, ys = vids[order], ys[order]
    uniq, starts = np.unique(vs, return_index=True)
    imp = np.diff(np.append(starts, len(vs))).astype(np.float64)
    pos = np.add.reduceat(ys.astype(np.float64), starts) if len(vs) else np.zeros(0)
    gmean = float(pos.sum() / imp.sum()) if imp.sum() else 0.0
    smoothed = (pos + PRIOR * gmean) / (imp + PRIOR)
    return dict(zip(uniq.tolist(), smoothed.tolist(), strict=True)), gmean


def write_submission(path: str, users: np.ndarray, videos: np.ndarray,
                     scores: np.ndarray) -> None:
    """`row_id,user_id,video_id,score`, one line per evaluation row, in split order."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        fh.writelines(
            "%d,%d,%d,%.10g\n" % (i, u, v, s)
            for i, (u, v, s) in enumerate(zip(users, videos, scores, strict=True))
        )


def main(argv=None) -> int:
    a = parse_args(argv)
    eval_split = SPLIT_ALIASES[a.split.lower()]
    started = time.time()

    df = load_logs(resolve_data_dir(a.data_dir))

    # Contract §1.2: train on train only for `--split val`; on train + validation for
    # `--split test`. The test period's labels are never read either way.
    fit_names = ("train",) if eval_split == "valid" else ("train", "valid")
    train = pd.concat([slice_split(df, n) for n in fit_names], ignore_index=True)
    n_users_all = int(train["user_id"].nunique())
    train = subsample_users(train, a.subsample, a.seed)

    table, gmean = fit_popularity(train)

    ev = slice_split(df, eval_split)
    eval_users = ev["user_id"].to_numpy()
    eval_videos = ev["video_id"].to_numpy()
    scores = np.array([table.get(int(v), gmean) for v in eval_videos], dtype=np.float64)
    train_seconds = time.time() - started

    os.makedirs(a.out_dir, exist_ok=True)
    write_submission(os.path.join(a.out_dir, "submission.csv"),
                     eval_users, eval_videos, scores)

    print("RESULT_JSON " + json.dumps({
        "n_rows": int(len(scores)),
        "train_seconds": round(float(train_seconds), 3),
        "notes": (
            "item popularity, prior=%g, fit on %s (%d rows, %d/%d users), "
            "%d videos seen, %d eval rows fell back to the global mean %.6f"
            % (PRIOR, "+".join(fit_names), len(train),
               int(train["user_id"].nunique()), n_users_all, len(table),
               int(sum(1 for v in eval_videos if int(v) not in table)), gmean)
        ),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
