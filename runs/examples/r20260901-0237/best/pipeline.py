import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
LATER_LOG = "log_standard_4_22_to_5_08_pure.csv"
VIDEO_FILE = "video_features_basic_pure.csv"
FEATURES = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
CAT_FEATURES = FEATURES[:]


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
    later = read_log(os.path.join(data_dir, LATER_LOG), mode == "val")
    if mode == "val":
        evaluation = later.loc[(later["date"] >= 20220422) & (later["date"] <= 20220428)].copy()
        fitting = train
    else:
        valid_for_fit = read_log(os.path.join(data_dir, LATER_LOG), True)
        valid_for_fit = valid_for_fit.loc[
            (valid_for_fit["date"] >= 20220422) & (valid_for_fit["date"] <= 20220428)
        ].copy()
        evaluation = later.loc[(later["date"] >= 20220429) & (later["date"] <= 20220508)].copy()
        fitting = pd.concat([train, valid_for_fit], axis=0, ignore_index=True)
    evaluation = evaluation.reset_index(drop=True)
    evaluation["row_id"] = np.arange(len(evaluation), dtype=np.int64)
    return fitting.reset_index(drop=True), evaluation


def add_author(frame, video):
    original_n = len(frame)
    work = frame.copy()
    work["_join_order"] = np.arange(original_n, dtype=np.int64)
    joined = work.merge(video, on="video_id", how="left", sort=False, validate="many_to_one")
    if len(joined) != original_n:
        raise RuntimeError("video join changed row count")
    joined = joined.sort_values("_join_order", kind="stable").drop(columns=["_join_order"])
    if len(joined) != original_n:
        raise RuntimeError("video join restoration failed")
    return joined


def duration_edges(values):
    x = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 17)))
    if len(edges) < 2:
        return np.array([-1.0, 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def prepare_features(fitting, evaluation, data_dir):
    video = pd.read_csv(os.path.join(data_dir, VIDEO_FILE), usecols=["video_id", "author_id"])
    if video["video_id"].duplicated().any():
        raise RuntimeError("video feature table is not one row per video")
    fitting = add_author(fitting, video)
    evaluation = add_author(evaluation, video)
    edges = duration_edges(fitting["duration_ms"])
    for frame in (fitting, evaluation):
        duration = pd.to_numeric(frame["duration_ms"], errors="coerce").fillna(0).to_numpy(dtype=float)
        frame["duration_bucket"] = np.digitize(duration, edges[1:-1], right=True).astype(np.int32)
    for col in CAT_FEATURES:
        values = fitting[col].astype("string").fillna("__MISSING__")
        unique = pd.Index(values.unique())
        mapping = {value: i for i, value in enumerate(unique)}
        fitting[col] = values.map(mapping).astype(np.int32)
        ev_values = evaluation[col].astype("string").fillna("__MISSING__")
        evaluation[col] = ev_values.map(mapping).fillna(-1).astype(np.int32)
    return fitting, evaluation


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


def ranked_dataset(frame):
    ordered = frame.sort_values("user_id", kind="stable").reset_index(drop=True)
    group = ordered.groupby("user_id", sort=False).size().to_numpy(dtype=np.int32)
    ds = lgb.Dataset(
        ordered[FEATURES],
        label=ordered["long_view"].astype(float),
        group=group,
        categorical_feature=CAT_FEATURES,
        free_raw_data=False,
    )
    return ordered, ds


def fit_predict(fitting, evaluation, mode, seed):
    _, dtrain = ranked_dataset(fitting)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "lambda_l2": 10.0,
        "lambdarank_truncation_level": 10,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": 4,
    }
    if mode == "val":
        _, dvalid = ranked_dataset(evaluation)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        iteration = booster.best_iteration
    else:
        booster = lgb.train(params, dtrain, num_boost_round=140, callbacks=[lgb.log_evaluation(0)])
        iteration = 140
    return np.asarray(booster.predict(evaluation[FEATURES], num_iteration=iteration), dtype=float)


def main():
    started = time.time()
    args = parse_args()
    fitting, evaluation = load_splits(args.data_dir, args.split)
    output_ids = evaluation[["row_id", "user_id", "video_id"]].copy()
    fitting, evaluation = prepare_features(fitting, evaluation, args.data_dir)
    fitting = maybe_subsample_users(fitting, args.subsample, args.seed)
    if fitting.empty:
        raise RuntimeError("no training rows after user-group subsampling")
    scores = fit_predict(fitting, evaluation, args.split, args.seed)
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
        "notes": "LightGBM LambdaRank official five fields; increased L2 regularisation to 10",
    }))


if __name__ == "__main__":
    main()
