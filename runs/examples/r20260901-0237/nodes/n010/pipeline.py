import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
LATER_LOG = "log_standard_4_22_to_5_08_pure.csv"
VIDEO_FILE = "video_features_basic_pure.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subsample", type=float, default=None)
    return p.parse_args()


def read_log(path):
    return pd.read_csv(path, dtype={"user_id": str, "video_id": str, "tab": str})


def load_splits(data_dir):
    train = read_log(os.path.join(data_dir, TRAIN_LOG))
    later = read_log(os.path.join(data_dir, LATER_LOG))
    dates = pd.to_numeric(later["date"], errors="raise")
    valid = later.loc[(dates >= 20220422) & (dates <= 20220428)].copy()
    test = later.loc[(dates >= 20220429) & (dates <= 20220508)].copy()
    return train, valid, test


def add_author(frame, author_lookup):
    out = frame.copy()
    n = len(out)
    out["author_id"] = out["video_id"].map(author_lookup).fillna("__UNKNOWN_AUTHOR__").astype(str)
    assert len(out) == n
    return out


def duration_bucket(values):
    x = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.minimum(np.floor(np.log2(np.maximum(x, 0.0) / 1000.0 + 1.0)).astype(np.int32), 20).astype(str)


def make_features(train, ev):
    tr = train.copy()
    te = ev.copy()
    tr["dur_bucket"] = duration_bucket(tr["duration_ms"])
    te["dur_bucket"] = duration_bucket(te["duration_ms"])
    fields = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
    blocks_train, blocks_ev = [], []
    offset = 0
    for field in fields:
        a = tr[field].fillna("__MISSING__").astype(str)
        b = te[field].fillna("__MISSING__").astype(str)
        vocabulary = pd.Index(a.unique())
        ca = vocabulary.get_indexer(a) + 1
        cb = vocabulary.get_indexer(b) + 1
        blocks_train.append((ca + offset).astype(np.int64))
        blocks_ev.append((cb + offset).astype(np.int64))
        offset += len(vocabulary) + 1
    return np.column_stack(blocks_train), np.column_stack(blocks_ev), offset


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, k=16):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(n_features, 1)
        self.factors = nn.Embedding(n_features, k)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(self, x):
        lin = self.linear(x).sum(dim=1).squeeze(1)
        v = self.factors(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (summed.square() - v.square().sum(dim=1)).sum(dim=1)
        return self.bias + lin + interaction


def choose_users(frame, fraction, seed):
    if fraction is None or fraction >= 1.0:
        return frame
    if fraction <= 0.0:
        raise ValueError("--subsample must be in (0, 1]")
    users = frame["user_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    keep_n = max(1, int(np.ceil(len(users) * fraction)))
    keep = set(rng.choice(users, size=keep_n, replace=False).tolist())
    return frame.loc[frame["user_id"].isin(keep)].copy()


def predict(model, x, batch_size=65536):
    model.eval()
    ans = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            z = torch.from_numpy(x[start:start + batch_size])
            ans.append(model(z).cpu().numpy())
    return np.concatenate(ans) if ans else np.empty(0, dtype=np.float64)


def eligible_groups(users, y):
    """Return positive and negative row-index arrays for users having both labels."""
    groups = []
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    for idx in np.split(order, cuts):
        pos = idx[y[idx] > 0.5]
        neg = idx[y[idx] <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos.astype(np.int64), neg.astype(np.int64)))
    return groups


def train_fm_bpr(x, y, users, seed, epochs):
    torch.manual_seed(seed)
    model = FactorizationMachine(int(x.max()) + 1, k=16)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    tx = torch.from_numpy(x)
    groups = eligible_groups(users, y)
    if not groups:
        raise RuntimeError("no users with both positive and negative training impressions")
    rng = np.random.default_rng(seed)
    batch = 8192
    for _epoch in range(epochs):
        # Each eligible positive appears once. Consequently a user's pair mass is its
        # positive count, rather than one equal vote per user.
        pos_parts = []
        neg_parts = []
        for pos, neg in groups:
            pos_parts.append(pos)
            neg_parts.append(neg[rng.integers(0, len(neg), size=len(pos))])
        pos_idx = np.concatenate(pos_parts)
        neg_idx = np.concatenate(neg_parts)
        perm = rng.permutation(len(pos_idx))
        pos_idx = pos_idx[perm]
        neg_idx = neg_idx[perm]
        for start in range(0, len(pos_idx), batch):
            p = torch.from_numpy(pos_idx[start:start + batch])
            n = torch.from_numpy(neg_idx[start:start + batch])
            margin = model(tx[p]) - model(tx[n])
            loss = F.softplus(-margin).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def main():
    started = time.time()
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    train, valid, test = load_splits(args.data_dir)
    if args.split == "val":
        fit_raw, evaluation = train, valid
    else:
        fit_raw, evaluation = pd.concat([train, valid], axis=0, ignore_index=True), test
    evaluation = evaluation.copy()
    evaluation["row_id"] = np.arange(len(evaluation), dtype=np.int64)

    video = pd.read_csv(os.path.join(args.data_dir, VIDEO_FILE), dtype={"video_id": str, "author_id": str})
    video = video[["video_id", "author_id"]].drop_duplicates("video_id", keep="first")
    assert video["video_id"].is_unique
    author_lookup = video.set_index("video_id")["author_id"]

    fit_raw = choose_users(fit_raw, args.subsample, args.seed)
    labels = pd.to_numeric(fit_raw["long_view"], errors="coerce")
    # Missing labels cannot form a supervised comparison. This only affects fitting rows.
    fit_raw = fit_raw.loc[labels.notna()].copy()
    y = labels.loc[labels.notna()].to_numpy(dtype=np.float32)
    fit = add_author(fit_raw, author_lookup)
    ev = add_author(evaluation, author_lookup)
    x_train, x_eval, _ = make_features(fit, ev)
    model = train_fm_bpr(x_train, y, fit["user_id"].to_numpy(), args.seed, epochs=40)
    scores = predict(model, x_eval)
    if len(scores) != len(evaluation) or not np.isfinite(scores).all():
        raise RuntimeError("invalid prediction vector")

    result = evaluation[["row_id", "user_id", "video_id"]].copy()
    result["score"] = scores
    result = result.sort_values("row_id", kind="stable")
    assert len(result) == len(evaluation)
    os.makedirs(args.out_dir, exist_ok=True)
    result.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False,
                  columns=["row_id", "user_id", "video_id", "score"], float_format="%.8f")
    notes = "five-field k=16 FM with positive-count-weighted within-user BPR; train-only" if args.split == "val" else "five-field k=16 FM with positive-count-weighted within-user BPR; train+validation"
    print("RESULT_JSON " + json.dumps({"n_rows": int(len(result)), "train_seconds": round(time.time() - started, 3), "notes": notes}))


if __name__ == "__main__":
    main()
