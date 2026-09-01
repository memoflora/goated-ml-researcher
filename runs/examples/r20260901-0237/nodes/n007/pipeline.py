import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
LATER_LOG = "log_standard_4_22_to_5_08_pure.csv"
VIDEO_FILE = "video_features_basic_pure.csv"
FEATURES = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsample", type=float, default=None)
    return parser.parse_args()


def read_log(path, include_label):
    columns = ["user_id", "video_id", "date", "tab", "duration_ms"]
    if include_label:
        columns.append("long_view")
    return pd.read_csv(path, usecols=columns)


def load_splits(data_dir, mode):
    train = read_log(os.path.join(data_dir, TRAIN_LOG), True)
    # In test mode this deliberately does not request long_view from the later log.
    later = read_log(os.path.join(data_dir, LATER_LOG), mode == "val")
    if mode == "val":
        evaluation = later.loc[(later["date"] >= 20220422) & (later["date"] <= 20220428)].copy()
        fitting = train
    else:
        # Labels are read only for the validation-date records which are permitted fitting data.
        later_labeled = read_log(os.path.join(data_dir, LATER_LOG), True)
        valid_for_fit = later_labeled.loc[
            (later_labeled["date"] >= 20220422) & (later_labeled["date"] <= 20220428)
        ].copy()
        evaluation = later.loc[(later["date"] >= 20220429) & (later["date"] <= 20220508)].copy()
        fitting = pd.concat([train, valid_for_fit], axis=0, ignore_index=True)
    evaluation = evaluation.reset_index(drop=True)
    evaluation["row_id"] = np.arange(len(evaluation), dtype=np.int64)
    return fitting.reset_index(drop=True), evaluation


def add_author(frame, video):
    n = len(frame)
    work = frame.copy()
    work["_join_order"] = np.arange(n, dtype=np.int64)
    result = work.merge(video, on="video_id", how="left", sort=False, validate="many_to_one")
    if len(result) != n:
        raise RuntimeError("video join changed row count")
    result = result.sort_values("_join_order", kind="stable").drop(columns=["_join_order"])
    if len(result) != n:
        raise RuntimeError("video join restoration failed")
    return result


def duration_edges(values):
    x = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 17)))
    if len(edges) < 2:
        return np.array([-1.0, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def prepare_features(fitting, evaluation, data_dir):
    video = pd.read_csv(os.path.join(data_dir, VIDEO_FILE), usecols=["video_id", "author_id"])
    if video["video_id"].duplicated().any():
        raise RuntimeError("video feature table is not one row per video")
    fitting = add_author(fitting, video)
    evaluation = add_author(evaluation, video)
    edges = duration_edges(fitting["duration_ms"])
    for frame in (fitting, evaluation):
        values = pd.to_numeric(frame["duration_ms"], errors="coerce").fillna(0).to_numpy(dtype=float)
        frame["duration_bucket"] = np.digitize(values, edges[1:-1], right=True).astype(np.int32)
    # Zero is reserved for unseen/missing values.  Offsets make field values disjoint.
    offsets, offset = [], 0
    for col in FEATURES:
        train_values = fitting[col].astype("string").fillna("__MISSING__")
        categories = pd.Index(train_values.unique())
        mapping = {v: i + 1 for i, v in enumerate(categories)}
        fitting[col] = train_values.map(mapping).astype(np.int64) + offset
        ev_values = evaluation[col].astype("string").fillna("__MISSING__")
        evaluation[col] = ev_values.map(mapping).fillna(0).astype(np.int64) + offset
        offsets.append(offset)
        offset += len(categories) + 1
    return fitting, evaluation, offset


def maybe_subsample_users(frame, fraction, seed):
    if fraction is None or fraction >= 1.0:
        return frame
    if fraction <= 0.0:
        raise ValueError("--subsample must be in (0, 1]")
    users = frame["user_id"].drop_duplicates().to_numpy()
    n_keep = max(1, int(np.ceil(len(users) * fraction)))
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(users, size=n_keep, replace=False).tolist())
    return frame.loc[frame["user_id"].isin(keep)].copy()


class FactorizationMachine(torch.nn.Module):
    def __init__(self, n_features, dim=16):
        super().__init__()
        self.linear = torch.nn.Embedding(n_features, 1)
        self.embedding = torch.nn.Embedding(n_features, dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * ((summed * summed) - (v * v).sum(dim=1)).sum(dim=1)
        return self.linear(x).squeeze(1).sum(dim=1) + interaction


def make_positive_pairs(frame):
    labels = pd.to_numeric(frame["long_view"], errors="coerce")
    usable = labels.notna().to_numpy()
    frame = frame.loc[usable].copy()
    labels = labels.loc[usable].to_numpy(dtype=float)
    order = np.argsort(frame["user_id"].to_numpy(), kind="stable")
    users = frame["user_id"].to_numpy()[order]
    y = labels[order] > 0.5
    positive, negative_sets = [], []
    starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(order)]
    for a, b in zip(starts[:-1], starts[1:]):
        group_rows = order[a:b]
        pos = group_rows[y[a:b]]
        neg = group_rows[~y[a:b]]
        if len(pos) and len(neg):
            positive.append(pos)
            negative_sets.extend([neg] * len(pos))
    if not positive:
        raise RuntimeError("no users with both positive and negative fitting impressions")
    return np.concatenate(positive).astype(np.int64), negative_sets


def fit_predict(fitting, evaluation, seed, n_feature_ids):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_x = fitting[FEATURES].to_numpy(dtype=np.int64, copy=True)
    eval_x = evaluation[FEATURES].to_numpy(dtype=np.int64, copy=True)
    positives, negative_sets = make_positive_pairs(fitting)
    model = FactorizationMachine(n_feature_ids, dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-5)
    batch_size = 4096
    # One negative for every positive makes pair sampling positive-count weighted.
    for _epoch in range(8):
        negs = np.fromiter((choices[rng.integers(len(choices))] for choices in negative_sets),
                           dtype=np.int64, count=len(positives))
        perm = rng.permutation(len(positives))
        for begin in range(0, len(perm), batch_size):
            take = perm[begin:begin + batch_size]
            xp = torch.from_numpy(train_x[positives[take]])
            xn = torch.from_numpy(train_x[negs[take]])
            optimizer.zero_grad(set_to_none=True)
            loss = -torch.nn.functional.logsigmoid(model(xp) - model(xn)).mean()
            loss.backward()
            optimizer.step()
    model.eval()
    answer = np.empty(len(eval_x), dtype=np.float64)
    with torch.no_grad():
        for begin in range(0, len(eval_x), 8192):
            answer[begin:begin + 8192] = model(torch.from_numpy(eval_x[begin:begin + 8192])).numpy()
    return answer


def main():
    started = time.time()
    args = parse_args()
    fitting, evaluation = load_splits(args.data_dir, args.split)
    output_ids = evaluation[["row_id", "user_id", "video_id"]].copy()
    fitting, evaluation, n_feature_ids = prepare_features(fitting, evaluation, args.data_dir)
    fitting = maybe_subsample_users(fitting, args.subsample, args.seed)
    if fitting.empty:
        raise RuntimeError("no training rows after user-group subsampling")
    scores = fit_predict(fitting, evaluation, args.seed, n_feature_ids)
    if len(scores) != len(evaluation) or not np.isfinite(scores).all():
        raise RuntimeError("model produced invalid predictions")
    output = output_ids.copy()
    output["score"] = scores
    output = output.sort_values("row_id", kind="stable")
    if len(output) != len(evaluation) or not np.array_equal(output["row_id"].to_numpy(), np.arange(len(output))):
        raise RuntimeError("evaluation row order was changed")
    os.makedirs(args.out_dir, exist_ok=True)
    output.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False, float_format="%.10g")
    print("RESULT_JSON " + json.dumps({
        "n_rows": int(len(output)),
        "train_seconds": round(time.time() - started, 3),
        "notes": "positive-count-weighted within-user BPR factorization machine on five official fields",
    }))


if __name__ == "__main__":
    main()
