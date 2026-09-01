import argparse
import json
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb


TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
LATER_LOG = "log_standard_4_22_to_5_08_pure.csv"
VIDEO_FILE = "video_features_basic_pure.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subsample", type=float, default=None)
    return p.parse_args()


def read_log(path):
    cols = ["user_id", "video_id", "date", "hourmin", "time_ms", "tab", "duration_ms", "long_view"]
    return pd.read_csv(path, usecols=cols, dtype={"user_id": "string", "video_id": "string", "date": "int32", "hourmin": "int32", "time_ms": "int64", "tab": "string", "duration_ms": "float32", "long_view": "Int8"})


def load_splits(data_dir, split):
    train = read_log(os.path.join(data_dir, TRAIN_LOG))
    later = read_log(os.path.join(data_dir, LATER_LOG))
    valid = later[(later.date >= 20220422) & (later.date <= 20220428)].copy()
    test = later[(later.date >= 20220429) & (later.date <= 20220508)].copy()
    if split == "val":
        return train.reset_index(drop=True), valid.reset_index(drop=True)
    fit = pd.concat([train, valid], ignore_index=True)
    return fit, test.reset_index(drop=True)


def apply_subsample(train, fraction, seed):
    if fraction is None or fraction >= 1.0:
        return train
    if fraction <= 0.0:
        raise ValueError("--subsample must be positive")
    users = train["user_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    keep_n = max(1, int(np.ceil(len(users) * fraction)))
    keep = set(rng.choice(users, size=keep_n, replace=False).tolist())
    return train[train["user_id"].isin(keep)].copy().reset_index(drop=True)


def add_author(train, ev, data_dir):
    basic = pd.read_csv(os.path.join(data_dir, VIDEO_FILE), usecols=["video_id", "author_id"], dtype={"video_id": "string", "author_id": "string"})
    author_map = basic.set_index("video_id")["author_id"]
    for df in (train, ev):
        df["author_id"] = df["video_id"].map(author_map).fillna("__missing__").astype("string")
    return train, ev


def add_exposure_features(train, ev):
    both = pd.concat([train.assign(_part=0), ev.assign(_part=1)], ignore_index=True)
    both["_seq"] = np.arange(len(both), dtype=np.int64)
    ordered = both.sort_values(["time_ms", "_seq"], kind="stable").copy()
    ordered["user_seen_log"] = np.log1p(ordered.groupby("user_id", sort=False).cumcount()).astype("float32")
    ordered["video_seen_log"] = np.log1p(ordered.groupby("video_id", sort=False).cumcount()).astype("float32")
    ordered["uv_seen_log"] = np.log1p(ordered.groupby(["user_id", "video_id"], sort=False).cumcount()).astype("float32")
    previous_user = ordered.groupby("user_id", sort=False)["time_ms"].shift(1)
    previous_uv = ordered.groupby(["user_id", "video_id"], sort=False)["time_ms"].shift(1)
    ordered["user_recency_log"] = np.log1p(((ordered["time_ms"] - previous_user).clip(lower=0).fillna(1.0)) / 60000.0).astype("float32")
    ordered["uv_recency_log"] = np.log1p(((ordered["time_ms"] - previous_uv).clip(lower=0).fillna(1.0)) / 60000.0).astype("float32")
    ordered = ordered.sort_values("_seq", kind="stable").drop(columns=["_seq"])
    a = ordered[ordered._part == 0].drop(columns=["_part"]).reset_index(drop=True)
    b = ordered[ordered._part == 1].drop(columns=["_part"]).reset_index(drop=True)
    return a, b


def add_temporal_rates(train, ev):
    keys = {
        "user": train["user_id"],
        "video": train["video_id"],
        "author": train["author_id"],
        "tab": train["tab"].fillna("__missing__"),
        "uv": train["user_id"].astype(str) + "|" + train["video_id"].astype(str),
    }
    evkeys = {
        "user": ev["user_id"],
        "video": ev["video_id"],
        "author": ev["author_id"],
        "tab": ev["tab"].fillna("__missing__"),
        "uv": ev["user_id"].astype(str) + "|" + ev["video_id"].astype(str),
    }
    states_sum = {name: pd.Series(dtype="float64") for name in keys}
    states_n = {name: pd.Series(dtype="float64") for name in keys}
    for name in keys:
        train[name + "_rate"] = np.float32(0.33)
        train[name + "_rate_count"] = np.float32(0.0)
    for day in np.sort(train["date"].unique()):
        idx = train.index[train["date"] == day]
        total_n = sum(float(x.sum()) for x in states_n.values())
        total_s = sum(float(x.sum()) for x in states_sum.values())
        prior = total_s / total_n if total_n > 0 else 0.33
        for name in keys:
            k = keys[name].loc[idx]
            n = k.map(states_n[name]).fillna(0.0)
            s = k.map(states_sum[name]).fillna(0.0)
            smooth = 20.0 if name != "uv" else 8.0
            train.loc[idx, name + "_rate"] = ((s + smooth * prior) / (n + smooth)).astype("float32")
            train.loc[idx, name + "_rate_count"] = np.log1p(n).astype("float32")
        y = train.loc[idx, "long_view"].astype("float64")
        for name in keys:
            k = keys[name].loc[idx]
            day_sum = y.groupby(k, sort=False).sum()
            day_n = y.groupby(k, sort=False).count().astype("float64")
            states_sum[name] = states_sum[name].add(day_sum, fill_value=0.0)
            states_n[name] = states_n[name].add(day_n, fill_value=0.0)
    total_n = sum(float(x.sum()) for x in states_n.values())
    total_s = sum(float(x.sum()) for x in states_sum.values())
    prior = total_s / total_n if total_n else 0.33
    for name in keys:
        n = evkeys[name].map(states_n[name]).fillna(0.0)
        s = evkeys[name].map(states_sum[name]).fillna(0.0)
        smooth = 20.0 if name != "uv" else 8.0
        ev[name + "_rate"] = ((s + smooth * prior) / (n + smooth)).astype("float32")
        ev[name + "_rate_count"] = np.log1p(n).astype("float32")
    return train, ev


def make_matrix(train, ev):
    all_ids = ["user_id", "video_id", "author_id", "tab"]
    for col in all_ids:
        categories = pd.Index(train[col].fillna("__missing__").unique())
        lookup = pd.Series(np.arange(len(categories), dtype=np.int32), index=categories)
        train[col + "_cat"] = train[col].fillna("__missing__").map(lookup).fillna(-1).astype("int32")
        ev[col + "_cat"] = ev[col].fillna("__missing__").map(lookup).fillna(-1).astype("int32")
    for df in (train, ev):
        duration = df["duration_ms"].fillna(0).clip(0, 1200000)
        df["duration_log"] = np.log1p(duration).astype("float32")
        df["duration_bucket"] = (np.log1p(duration) // 0.7).clip(0, 30).astype("int32")
        df["hour"] = (df["hourmin"].fillna(0) // 100).clip(0, 23).astype("int32")
    rate_cols = [c for c in train.columns if c.endswith("_rate") or c.endswith("_rate_count")]
    exposure = ["user_seen_log", "video_seen_log", "uv_seen_log", "user_recency_log", "uv_recency_log", "duration_log"]
    cats = [c + "_cat" for c in all_ids] + ["duration_bucket", "hour"]
    cols = rate_cols + exposure + cats
    return train[cols], ev[cols], cats


def fit_predict(train, ev, args):
    xtr, xev, cats = make_matrix(train, ev)
    y = train["long_view"].astype(int)
    params = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.045,
        "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.85,
        "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 3.0,
        "verbosity": -1, "seed": args.seed, "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed, "data_random_seed": args.seed, "num_threads": 1,
    }
    cat_indices = [xtr.columns.get_loc(c) for c in cats]
    if args.split == "val":
        dtrain = lgb.Dataset(xtr, label=y, categorical_feature=cat_indices, free_raw_data=False)
        labeled = ev["long_view"].notna().to_numpy()
        if labeled.any():
            dval = lgb.Dataset(xev.loc[labeled], label=ev.loc[labeled, "long_view"].astype(int), categorical_feature=cat_indices, reference=dtrain, free_raw_data=False)
            model = lgb.train(params, dtrain, num_boost_round=350, valid_sets=[dval], callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(0)])
        else:
            model = lgb.train(params, dtrain, num_boost_round=350, callbacks=[lgb.log_evaluation(0)])
    else:
        dtrain = lgb.Dataset(xtr, label=y, categorical_feature=cat_indices)
        model = lgb.train(params, dtrain, num_boost_round=140, callbacks=[lgb.log_evaluation(0)])
    return model.predict(xev, num_iteration=model.best_iteration or model.current_iteration())


def main():
    started = time.time()
    args = parse_args()
    train, ev = load_splits(args.data_dir, args.split)
    ev = ev.copy()
    ev["row_id"] = np.arange(len(ev), dtype=np.int64)
    train = apply_subsample(train, args.subsample, args.seed)
    train, ev = add_author(train, ev, args.data_dir)
    train, ev = add_exposure_features(train, ev)
    train, ev = add_temporal_rates(train, ev)
    pred = np.nan_to_num(fit_predict(train, ev, args), nan=0.0, posinf=1e6, neginf=-1e6)
    out = pd.DataFrame({"row_id": ev["row_id"], "user_id": ev["user_id"], "video_id": ev["video_id"], "score": pred})
    out = out.sort_values("row_id", kind="stable")
    os.makedirs(args.out_dir, exist_ok=True)
    out.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False)
    print("RESULT_JSON " + json.dumps({"n_rows": int(len(out)), "train_seconds": round(time.time() - started, 3), "notes": "temporal smoothed CTR/count/recency LightGBM"}))


if __name__ == "__main__":
    main()
