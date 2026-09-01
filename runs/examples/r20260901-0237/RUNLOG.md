# Run log — `r20260901-0237`

Per-iteration record required by the Starter Kit: the hypothesis the agent formed,
the code diff it applied, the metrics that came back, and any error it recovered
from. Generated from the run directory by `python -m orchestrator.runlog`.

- task: `kuairand-pure` · mode: `dev` · model: `gpt-5.6-terra`
- commit: `16571db249af1464563160d4e2236e6609618b0b` · subsample: `1.0`
- limits: 12 iterations, 1800s wall-clock, conv_eps=0.002 conv_n=4 explore_after=2

## Manual interventions

**0** — every iteration below ran unattended, from launch to the final summary.

## Iterations

### Iteration 1 — `n000` (draft)

**Hypothesis** — what it intended to try, and why

> The official baseline should reach roughly validation primary 0.6016 because the logged impression outcome has strong sparse user-by-video, user-by-author, and context interactions that are represented directly by a k=16 factorization machine; using only the five prescribed categorical fields prevents a first draft from confusing untested side-feature effects with basic pipeline correctness.

**Plan**

- Load the two standard logs in documented file/date order and assign evaluation row positions before any feature work.
- Join only the one-to-one video-to-author lookup, checking that it cannot alter impression count or order.
- Encode the official five categorical fields (user, video, author, tab, bucketed duration) from the training portion and train a deterministic k=16 logistic FM in PyTorch.
- Use train-only fitting for validation, train-plus-validation fitting for test, and sample whole training-user groups when --subsample is supplied.
- Write finite sigmoid scores in restored evaluation row order with the required submission schema and one RESULT_JSON line.

**Ideas drawn on:** T0.reproduce-fm

**Code diff applied**

_new program (no parent to diff against) — 184 lines, full source at `nodes/n000/pipeline.py`_

**Resulting metrics**

| metric | value | vs baseline |
|---|---|---|
| gauc | 0.62551 | -0.04190 |
| ndcg@5 | 0.51751 | -0.01820 |
| primary | 0.57151 | -0.03010 |

### Iteration 2 — `n001` (draft)

**Hypothesis** — what it intended to try, and why

> The main limitation of the constant floor is that it ignores the highly repeatable item- and user–item-level long-view propensity in this logged exposure data. A leakage-safe, temporally accumulated set of smoothed CTR and exposure/recency features, combined with a modest GBDT that can cross those item signals with user and context identifiers, should reproduce or modestly exceed the 0.6016 validation primary (roughly 0.60–0.62) without embedding capacity; this claim would be false if the observed item propensities do not persist from the training dates into validation.

**Plan**

- Load the prescribed date splits in source-file order and assign evaluation row positions before any feature work.
- Join only the one-to-one video author lookup, then build exposure counts and last-exposure recencies from impressions available earlier in time.
- Build smoothed user, video, author, tab, and user-video long-view rates using only labels from strictly earlier calendar dates; validation/test labels are never used as features.
- Train a deterministic LightGBM binary scorer with native categorical identifiers and early stopping on held-out validation logloss; for test, refit on train plus validation using the selected tree count.
- Implement user-group subsampling only on the fitting history and always emit every evaluation row in its original order.

**Ideas drawn on:** T0.early-stop-on-primary, T0.item-pop-prior

**Code diff applied**

_new program (no parent to diff against) — 195 lines, full source at `nodes/n001/pipeline.py`_

**Resulting metrics**

_no metrics — this iteration did not produce a scored submission._

**Errors and recovery**

- `error` · class `runtime` — File "/Users/nabilrakaizaabror/Documents/GitHub/goated-ml-researcher/runs/r20260901-0237/nodes/n001/pipeline.py", line 28, in read_log
    return pd.read_csv(path, usecols=cols, dtype={"user_id": "string", "video_id": "string", "date": "int32", "hourmin": "int32", "time_ms": "int64", "tab": "string", "duration_ms": "float32", "long_view": "int8"})
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `recovery` · **schedule_debug** (repair 1/3)

### Iteration 3 — `n002` (debug from `n001`)

**Hypothesis** — what it intended to try, and why

> The runtime failure is caused by pandas being asked to parse `long_view` as a non-nullable NumPy `int8` even though that source column contains at least one missing value; changing only that input dtype to pandas' nullable `Int8` preserves the same values and downstream feature/model logic while allowing CSV parsing to complete, so validation primary should be unchanged apart from normal run noise.

**Plan**

- Change only `long_view` in `read_log` from NumPy `int8` to pandas nullable `Int8`, which accepts missing CSV values.
- Leave split construction, feature engineering, model configuration, row-id handling, and output unchanged.

**Code diff applied**

```diff
--- nodes/n001/pipeline.py
+++ nodes/n002/pipeline.py
@@ -25,7 +25,7 @@
 
 def read_log(path):
     cols = ["user_id", "video_id", "date", "hourmin", "time_ms", "tab", "duration_ms", "long_view"]
-    return pd.read_csv(path, usecols=cols, dtype={"user_id": "string", "video_id": "string", "date": "int32", "hourmin": "int32", "time_ms": "int64", "tab": "string", "duration_ms": "float32", "long_view": "int8"})
+    return pd.read_csv(path, usecols=cols, dtype={"user_id": "string", "video_id": "string", "date": "int32", "hourmin": "int32", "time_ms": "int64", "tab": "string", "duration_ms": "float32", "long_view": "Int8"})
 
 
 def load_splits(data_dir, split):
```

**Resulting metrics**

_no metrics — this iteration did not produce a scored submission._

**Errors and recovery**

- `error` · class `runtime` — File "/Users/nabilrakaizaabror/Documents/GitHub/goated-ml-researcher/runs/r20260901-0237/nodes/n002/pipeline.py", line 168, in fit_predict
    dval = lgb.Dataset(xev, label=ev["long_view"].astype(int), categorical_feature=cat_indices, reference=dtrain, free_raw_data=False)
                                  ~~~~~~~~~~~~~~~~~~~~~~^^^^^
ValueError: cannot convert NA to integer
- `recovery` · **schedule_debug** (repair 2/3)

### Iteration 4 — `n003` (debug from `n002`)

**Hypothesis** — what it intended to try, and why

> The validation interaction file is carrying missing `long_view` values for at least some evaluation impressions, so the failure is caused specifically by passing a nullable held-out label series to LightGBM rather than by the feature matrix. Restricting early-stopping monitoring to evaluation rows whose labels are actually present (and falling back to the existing maximum-round fit if none are present) should make validation execution succeed without changing training features, fitted training labels, prediction rows, or their order; score movement should be negligible when labels are complete and otherwise reflects only removal of unusable monitor rows.

**Plan**

- Keep split loading, features, LightGBM parameters, training data, and submission construction unchanged.
- In validation mode, build the LightGBM monitoring Dataset only from evaluation rows with non-missing `long_view`.
- Use the same early-stopping configuration when at least one labeled monitor row exists; if no evaluation labels are available, train for the already configured 350 rounds without an invalid monitoring Dataset.
- Continue predicting every evaluation row and preserve `row_id` order exactly.

**Code diff applied**

```diff
--- nodes/n002/pipeline.py
+++ nodes/n003/pipeline.py
@@ -161,12 +161,14 @@
     }
     cat_indices = [xtr.columns.get_loc(c) for c in cats]
     if args.split == "val":
-        # The supplied evaluation labels are legitimate validation-only early-stopping data.
+        # Held-out files can contain unlabeled impressions; only labeled rows can monitor loss.
         dtrain = lgb.Dataset(xtr, label=y, categorical_feature=cat_indices, free_raw_data=False)
-        # Reserve a deterministic tail of training dates for stopping, avoiding use of eval labels in fitting.
-        # Evaluation labels are still used by LightGBM only as validation monitoring in this development mode.
-        dval = lgb.Dataset(xev, label=ev["long_view"].astype(int), categorical_feature=cat_indices, reference=dtrain, free_raw_data=False)
-        model = lgb.train(params, dtrain, num_boost_round=350, valid_sets=[dval], callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(0)])
+        labeled = ev["long_view"].notna().to_numpy()
+        if labeled.any():
+            dval = lgb.Dataset(xev.loc[labeled], label=ev.loc[labeled, "long_view"].astype(int), categorical_feature=cat_indices, reference=dtrain, free_raw_data=False)
+            model = lgb.train(params, dtrain, num_boost_round=350, valid_sets=[dval], callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(0)])
+        else:
+            model = lgb.train(params, dtrain, num_boost_round=350, callbacks=[lgb.log_evaluation(0)])
     else:
         dtrain = lgb.Dataset(xtr, label=y, categorical_feature=cat_indices)
         model = lgb.train(params, dtrain, num_boost_round=140, callbacks=[lgb.log_evaluation(0)])
```

**Resulting metrics**

_no metrics — this iteration did not produce a scored submission._

**Errors and recovery**

- `error` · class `contract` — submission failed validation: line 2: row_id '0.0' is not an integer — it must be the 0-based position of the row in the valid split (0 on this line)
- `recovery` · **schedule_debug** (repair 3/3)

### Iteration 5 — `n004` (debug from `n003`)

**Hypothesis** — what it intended to try, and why

> The absence of a traceback combined with a completed RESULT_JSON suggests a native-library shutdown or threading failure rather than a Python/data exception; LightGBM was the only compiled training component configured for parallel execution. Forcing its OpenMP and LightGBM execution to one thread should eliminate a thread-runtime crash while preserving the identical features, data, objective, and model hyperparameters, with no expected validation-score movement beyond normal noise.

**Plan**

- Set OMP_NUM_THREADS=1 before importing NumPy or LightGBM.
- Set LightGBM num_threads to 1 as the sole model configuration change.
- Keep split construction, feature engineering, training rounds, prediction, and ordered submission writing unchanged.

**Code diff applied**

```diff
--- nodes/n003/pipeline.py
+++ nodes/n004/pipeline.py
@@ -2,6 +2,8 @@
 import json
 import os
 import time
+
+os.environ["OMP_NUM_THREADS"] = "1"
 
 import numpy as np
 import pandas as pd
@@ -53,7 +55,6 @@
 
 def add_author(train, ev, data_dir):
     basic = pd.read_csv(os.path.join(data_dir, VIDEO_FILE), usecols=["video_id", "author_id"], dtype={"video_id": "string", "author_id": "string"})
-    # File is documented one row per video; a mapping avoids any merge-induced row reorder.
     author_map = basic.set_index("video_id")["author_id"]
     for df in (train, ev):
         df["author_id"] = df["video_id"].map(author_map).fillna("__missing__").astype("string")
@@ -97,7 +98,6 @@
     for name in keys:
         train[name + "_rate"] = np.float32(0.33)
         train[name + "_rate_count"] = np.float32(0.0)
-    # A day is the atomic label update, so no row sees its own (or same-day) target.
     for day in np.sort(train["date"].unique()):
         idx = train.index[train["date"] == day]
         total_n = sum(float(x.sum()) for x in states_n.values())
@@ -132,7 +132,6 @@
 def make_matrix(train, ev):
     all_ids = ["user_id", "video_id", "author_id", "tab"]
     for col in all_ids:
-        # Category codes are learned from fit data only; unknown evaluation identities are missing.
         categories = pd.Index(train[col].fillna("__missing__").unique())
         lookup = pd.Series(np.arange(len(categories), dtype=np.int32), index=categories)
         train[col + "_cat"] = train[col].fillna("__missing__").map(lookup).fillna(-1).astype("int32")
@@ -157,11 +156,10 @@
         "num_leaves": 31, "min_data_in_leaf": 80, "feature_fraction": 0.85,
         "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 3.0,
         "verbosity": -1, "seed": args.seed, "feature_fraction_seed": args.seed,
-        "bagging_seed": args.seed, "data_random_seed": args.seed, "num_threads": 4,
+        "bagging_seed": args.seed, "data_random_seed": args.seed, "num_threads": 1,
     }
     cat_indices = [xtr.columns.get_loc(c) for c in cats]
     if args.split == "val":
-        # Held-out files can contain unlabeled impressions; only labeled rows can monitor loss.
         dtrain = lgb.Dataset(xtr, label=y, categorical_feature=cat_indices, free_raw_data=False)
         labeled = ev["long_view"].notna().to_numpy()
         if labeled.any():
```

**Resulting metrics**

_no metrics — this iteration did not produce a scored submission._

**Errors and recovery**

- `error` · class `contract` — submission failed validation: line 2: row_id '0.0' is not an integer — it must be the 0-based position of the row in the valid split (0 on this line)

### Iteration 6 — `n005` (draft)

**Hypothesis** — what it intended to try, and why

> The limiting issue for a first usable KuaiRand model is that global pointwise probability fitting does not match the per-user ordering used by both GAUC and nDCG@5. I expect a deterministic LightGBM LambdaRank model trained on complete user impression lists, using the official user/video/author/tab/duration fields and early stopping on held-out nDCG@5, to reach roughly the published 0.6016 validation primary (within a few thousandths) because its loss directly rewards moving positive impressions above negatives within each user.

**Plan**

- Load only the two standard-exposure logs, construct the fixed date splits, and assign evaluation row_id before any feature join.
- Join the one-to-one video author lookup with an explicit row-count check, then use the official five ranking fields: user_id, video_id, author_id, tab, and a train-derived duration bucket.
- Encode categorical values from the fitting data only and train a deterministic LightGBM `lambdarank` model on user-contiguous impression groups.
- For validation, early-stop on LightGBM's validation nDCG@5; for test, fit on train plus validation for the selected fixed number of rounds without reading test labels.
- Write predictions back in the untouched evaluation row order and emit exactly one RESULT_JSON line.

**Ideas drawn on:** T1.listwise-softmax

**Code diff applied**

_new program (no parent to diff against) — 193 lines, full source at `nodes/n005/pipeline.py`_

**Resulting metrics**

_no metrics — this iteration did not produce a scored submission._

**Errors and recovery**

- `error` · class `contract` — submission failed validation: line 2 misaligned: submission has (0,477), valid row 0 is (0,7531). Emit the evaluation rows in the split's own order; (user_id, video_id) is not unique, so never merge or key on it
- `recovery` · **schedule_debug** (repair 1/3)

### Iteration 7 — `n006` (debug from `n005`)

**Hypothesis** — what it intended to try, and why

> The program reached its RESULT_JSON line, so the likely failure is downstream submission validation rather than LightGBM training: `prepare_features` overwrites `evaluation['user_id']` and `evaluation['video_id']` with LightGBM category codes, and those codes are then written as the submission identifiers. Preserving the original evaluation identifiers for output while continuing to use the same encoded feature frame should make the submission align with the loaded split and raise validation primary from rejection to the model's actual score.

**Plan**

- Save `row_id`, `user_id`, and `video_id` from the evaluation split immediately after loading and before feature encoding.
- Continue to encode the same evaluation frame and train exactly the same LambdaRank model.
- Build the submission from the preserved original identifier frame, attaching predictions by unchanged evaluation row position.
- Retain all existing split logic, model parameters, joins, checks, and output formatting.

**Code diff applied**

```diff
--- nodes/n005/pipeline.py
+++ nodes/n006/pipeline.py
@@ -83,8 +83,6 @@
     for frame in (fitting, evaluation):
         duration = pd.to_numeric(frame["duration_ms"], errors="coerce").fillna(0).to_numpy(dtype=float)
         frame["duration_bucket"] = np.digitize(duration, edges[1:-1], right=True).astype(np.int32)
-    # Codes are learned from fitting data only.  Unknown future identifiers become
-    # LightGBM missing categorical values rather than being assigned a misleading ID.
     for col in CAT_FEATURES:
         values = fitting[col].astype("string").fillna("__MISSING__")
         unique = pd.Index(values.unique())
@@ -108,7 +106,6 @@
 
 
 def ranked_dataset(frame):
-    # LambdaRank requires rows belonging to a query to be contiguous.
     ordered = frame.sort_values("user_id", kind="stable").reset_index(drop=True)
     group = ordered.groupby("user_id", sort=False).size().to_numpy(dtype=np.int32)
     ds = lgb.Dataset(
@@ -153,8 +150,6 @@
         )
         iteration = booster.best_iteration
     else:
-        # The held-out test label is never loaded.  This conservative fixed count is
-        # used after fitting on all development-period labels.
         booster = lgb.train(params, dtrain, num_boost_round=140, callbacks=[lgb.log_evaluation(0)])
         iteration = 140
     scores = booster.predict(evaluation[FEATURES], num_iteration=iteration)
@@ -167,6 +162,7 @@
     args = parse_args()
     DATA_DIR = args.data_dir
     fitting, evaluation = load_splits(args.data_dir, args.split)
+    output_ids = evaluation[["row_id", "user_id", "video_id"]].copy()
     fitting, evaluation = prepare_features(fitting, evaluation)
     fitting = maybe_subsample_users(fitting, args.subsample, args.seed)
     if fitting.empty:
@@ -174,7 +170,7 @@
     scores = fit_predict(fitting, evaluation, args.split, args.seed)
     if len(scores) != len(evaluation) or not np.isfinite(scores).all():
         raise RuntimeError("model produced invalid predictions")
-    output = evaluation[["row_id", "user_id", "video_id"]].copy()
+    output = output_ids.copy()
     output["score"] = scores
     output = output.sort_values("row_id", kind="stable")
     if len(output) != len(evaluation) or not np.array_equal(output["row_id"].to_numpy(), np.arange(len(output))):
```

**Resulting metrics**

| metric | value | vs baseline |
|---|---|---|
| gauc | 0.63971 | -0.02770 |
| ndcg@5 | 0.52404 | -0.01170 |
| primary | 0.58188 | -0.01970 |

### Iteration 8 — `n007` (improve from `n006`)

**Hypothesis** — what it intended to try, and why

> The current LambdaRank tree model is likely limited by its pointwise categorical splits rather than its use of grouped rows: it cannot efficiently represent the dense user-by-video/author/context crosses that determine an impression's order. Replacing it with a compact factorization-machine scorer trained directly with positive-count-weighted within-user BPR pairs should align the loss with GAUC while learning those crosses, and I expect a material recovery toward the 0.60 primary baseline (roughly +0.01 to +0.02 if the pairwise FM transfers).

**Plan**

- Replace only the ranking learner with a Torch factorization machine trained by BPR on positive/negative impressions from the same user.
- Sample one pair per eligible positive interaction per epoch; this makes users contribute in proportion to their positive count, as required by GAUC weighting.
- Retain the existing date split semantics, five official fields, train-only categorical fitting, duration bucketing, group-based subsampling, and row-id-preserving submission path.
- Use the validation labels only to train neither model nor select epochs; test mode fits on train plus validation and never reads test labels.

**Ideas drawn on:** T1.bpr-pairwise

**Code diff applied**

```diff
--- nodes/n006/pipeline.py
+++ nodes/n007/pipeline.py
@@ -1,18 +1,16 @@
 import argparse
-import csv
 import json
 import os
 import time
 
-import lightgbm as lgb
 import numpy as np
 import pandas as pd
+import torch
 
 TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
 LATER_LOG = "log_standard_4_22_to_5_08_pure.csv"
 VIDEO_FILE = "video_features_basic_pure.csv"
 FEATURES = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
-CAT_FEATURES = FEATURES[:]
 
 
 def parse_args():
@@ -34,14 +32,16 @@
 
 def load_splits(data_dir, mode):
     train = read_log(os.path.join(data_dir, TRAIN_LOG), True)
+    # In test mode this deliberately does not request long_view from the later log.
     later = read_log(os.path.join(data_dir, LATER_LOG), mode == "val")
     if mode == "val":
         evaluation = later.loc[(later["date"] >= 20220422) & (later["date"] <= 20220428)].copy()
         fitting = train
     else:
-        valid_for_fit = read_log(os.path.join(data_dir, LATER_LOG), True)
-        valid_for_fit = valid_for_fit.loc[
-            (valid_for_fit["date"] >= 20220422) & (valid_for_fit["date"] <= 20220428)
+        # Labels are read only for the validation-date records which are permitted fitting data.
+        later_labeled = read_log(os.path.join(data_dir, LATER_LOG), True)
+        valid_for_fit = later_labeled.loc[
+            (later_labeled["date"] >= 20220422) & (later_labeled["date"] <= 20220428)
         ].copy()
         evaluation = later.loc[(later["date"] >= 20220429) & (later["date"] <= 20220508)].copy()
         fitting = pd.concat([train, valid_for_fit], axis=0, ignore_index=True)
@@ -51,16 +51,16 @@
 
 
 def add_author(frame, video):
-    original_n = len(frame)
+    n = len(frame)
     work = frame.copy()
-    work["_join_order"] = np.arange(original_n, dtype=np.int64)
-    joined = work.merge(video, on="video_id", how="left", sort=False, validate="many_to_one")
-    if len(joined) != original_n:
+    work["_join_order"] = np.arange(n, dtype=np.int64)
+    result = work.merge(video, on="video_id", how="left", sort=False, validate="many_to_one")
+    if len(result) != n:
         raise RuntimeError("video join changed row count")
-    joined = joined.sort_values("_join_order", kind="stable").drop(columns=["_join_order"])
-    if len(joined) != original_n:
+    result = result.sort_values("_join_order", kind="stable").drop(columns=["_join_order"])
+    if len(result) != n:
         raise RuntimeError("video join restoration failed")
-    return joined
+    return result
 
 
 def duration_edges(values):
@@ -68,29 +68,32 @@
     edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, 17)))
     if len(edges) < 2:
         return np.array([-1.0, 1.0])
-    edges[0] = -np.inf
-    edges[-1] = np.inf
+    edges[0], edges[-1] = -np.inf, np.inf
     return edges
 
 
-def prepare_features(fitting, evaluation):
-    video = pd.read_csv(os.path.join(DATA_DIR, VIDEO_FILE), usecols=["video_id", "author_id"])
+def prepare_features(fitting, evaluation, data_dir):
+    video = pd.read_csv(os.path.join(data_dir, VIDEO_FILE), usecols=["video_id", "author_id"])
     if video["video_id"].duplicated().any():
         raise RuntimeError("video feature table is not one row per video")
     fitting = add_author(fitting, video)
     evaluation = add_author(evaluation, video)
     edges = duration_edges(fitting["duration_ms"])
     for frame in (fitting, evaluation):
-        duration = pd.to_numeric(frame["duration_ms"], errors="coerce").fillna(0).to_numpy(dtype=float)
-        frame["duration_bucket"] = np.digitize(duration, edges[1:-1], right=True).astype(np.int32)
-    for col in CAT_FEATURES:
-        values = fitting[col].astype("string").fillna("__MISSING__")
-        unique = pd.Index(values.unique())
-        mapping = {value: i for i, value in enumerate(unique)}
-        fitting[col] = values.map(mapping).astype(np.int32)
+        values = pd.to_numeric(frame["duration_ms"], errors="coerce").fillna(0).to_numpy(dtype=float)
+        frame["duration_bucket"] = np.digitize(values, edges[1:-1], right=True).astype(np.int32)
+    # Zero is reserved for unseen/missing values.  Offsets make field values disjoint.
+    offsets, offset = [], 0
+    for col in FEATURES:
+        train_values = fitting[col].astype("string").fillna("__MISSING__")
+        categories = pd.Index(train_values.unique())
+        mapping = {v: i + 1 for i, v in enumerate(categories)}
+        fitting[col] = train_values.map(mapping).astype(np.int64) + offset
         ev_values = evaluation[col].astype("string").fillna("__MISSING__")
-        evaluation[col] = ev_values.map(mapping).fillna(-1).astype(np.int32)
-    return fitting, evaluation
+        evaluation[col] = ev_values.map(mapping).fillna(0).astype(np.int64) + offset
+        offsets.append(offset)
+        offset += len(categories) + 1
+    return fitting, evaluation, offset
 
 
 def maybe_subsample_users(frame, fraction, seed):
@@ -105,69 +108,84 @@
     return frame.loc[frame["user_id"].isin(keep)].copy()
 
 
-def ranked_dataset(frame):
-    ordered = frame.sort_values("user_id", kind="stable").reset_index(drop=True)
-    group = ordered.groupby("user_id", sort=False).size().to_numpy(dtype=np.int32)
-    ds = lgb.Dataset(
```
_diff truncated at 120 of 264 lines; full source at `nodes/n007/pipeline.py`_

**Resulting metrics**

_no metrics — this iteration did not produce a scored submission._

**Errors and recovery**

- `error` · class `runtime` — File "/Users/nabilrakaizaabror/Documents/GitHub/goated-ml-researcher/runs/r20260901-0237/nodes/n007/pipeline.py", line 175, in fit_predict
    answer[begin:begin + 8192] = model(torch.from_numpy(eval_x[begin:begin + 8192])).numpy()
    ~~~~~~^^^^^^^^^^^^^^^^^^^^
ValueError: could not broadcast input array from shape (8192,8192) into shape (8192,)
- `recovery` · **schedule_debug** (repair 1/3)

### Iteration 9 — `n008` (debug from `n007`)

**Hypothesis** — what it intended to try, and why

> The runtime failure is caused by an unintended PyTorch broadcast in the FM first-order term: `squeeze(1)` does not remove the final singleton embedding dimension from a `[batch, 5, 1]` tensor, so the `[batch, 1]` linear result added to the `[batch]` interaction result becomes `[batch, batch]`. Replacing it with `squeeze(-1)` restores a one-dimensional `[batch]` model output, which should allow batched evaluation assignments to succeed without changing the model or training procedure.

**Plan**

- Change only the first-order FM output squeeze from dimension 1 to the final singleton dimension (`squeeze(-1)`).
- Keep all data loading, feature construction, sampling, optimization, prediction batching, and submission logic unchanged.
- The corrected forward pass returns one scalar score per input row, matching the destination slice shape.

**Code diff applied**

```diff
--- nodes/n007/pipeline.py
+++ nodes/n008/pipeline.py
@@ -120,7 +120,7 @@
         v = self.embedding(x)
         summed = v.sum(dim=1)
         interaction = 0.5 * ((summed * summed) - (v * v).sum(dim=1)).sum(dim=1)
-        return self.linear(x).squeeze(1).sum(dim=1) + interaction
+        return self.linear(x).squeeze(-1).sum(dim=1) + interaction
 
 
 def make_positive_pairs(frame):
```

**Resulting metrics**

| metric | value | vs baseline |
|---|---|---|
| gauc | 0.59391 | -0.07350 |
| ndcg@5 | 0.50410 | -0.03160 |
| primary | 0.54901 | -0.05260 |

### Iteration 10 — `n009` (improve from `n006`)

**Hypothesis** — what it intended to try, and why

> The current LambdaRank model is likely overfitting sparse, high-cardinality user/video categorical partitions: its validation-selected ranking is still far below the known five-field baseline, and the supplied evidence specifically identifies regularisation as remaining hyperparameter headroom whereas embedding capacity was flat. Increasing only the tree L2 penalty should make leaf scores less sensitive to transient train-user/item outcomes and improve date-forward within-user ordering by roughly 0.002–0.006 primary if this is the binding limitation.

**Plan**

- Keep split construction, joins, feature set, LambdaRank objective, early stopping, and submission order unchanged.
- Change exactly one modelling setting: increase LightGBM `lambda_l2` from 1.0 to 10.0.
- Retain deterministic seeds and four training threads so this is directly comparable to the incumbent.
- Continue to sample whole original user groups only when `--subsample` is supplied.

**Ideas drawn on:** T0.l2-and-lr-schedule

**Code diff applied**

```diff
--- nodes/n006/pipeline.py
+++ nodes/n009/pipeline.py
@@ -1,5 +1,4 @@
 import argparse
-import csv
 import json
 import os
 import time
@@ -73,8 +72,8 @@
     return edges
 
 
-def prepare_features(fitting, evaluation):
-    video = pd.read_csv(os.path.join(DATA_DIR, VIDEO_FILE), usecols=["video_id", "author_id"])
+def prepare_features(fitting, evaluation, data_dir):
+    video = pd.read_csv(os.path.join(data_dir, VIDEO_FILE), usecols=["video_id", "author_id"])
     if video["video_id"].duplicated().any():
         raise RuntimeError("video feature table is not one row per video")
     fitting = add_author(fitting, video)
@@ -119,7 +118,7 @@
 
 
 def fit_predict(fitting, evaluation, mode, seed):
-    ordered, dtrain = ranked_dataset(fitting)
+    _, dtrain = ranked_dataset(fitting)
     params = {
         "objective": "lambdarank",
         "metric": "ndcg",
@@ -129,7 +128,7 @@
         "min_data_in_leaf": 40,
         "feature_fraction": 1.0,
         "bagging_fraction": 1.0,
-        "lambda_l2": 1.0,
+        "lambda_l2": 10.0,
         "lambdarank_truncation_level": 10,
         "verbosity": -1,
         "seed": seed,
@@ -152,18 +151,15 @@
     else:
         booster = lgb.train(params, dtrain, num_boost_round=140, callbacks=[lgb.log_evaluation(0)])
         iteration = 140
-    scores = booster.predict(evaluation[FEATURES], num_iteration=iteration)
-    return np.asarray(scores, dtype=float)
+    return np.asarray(booster.predict(evaluation[FEATURES], num_iteration=iteration), dtype=float)
 
 
 def main():
-    global DATA_DIR
     started = time.time()
     args = parse_args()
-    DATA_DIR = args.data_dir
     fitting, evaluation = load_splits(args.data_dir, args.split)
     output_ids = evaluation[["row_id", "user_id", "video_id"]].copy()
-    fitting, evaluation = prepare_features(fitting, evaluation)
+    fitting, evaluation = prepare_features(fitting, evaluation, args.data_dir)
     fitting = maybe_subsample_users(fitting, args.subsample, args.seed)
     if fitting.empty:
         raise RuntimeError("no training rows after user-group subsampling")
@@ -177,11 +173,10 @@
         raise RuntimeError("evaluation row order was changed")
     os.makedirs(args.out_dir, exist_ok=True)
     output.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False, float_format="%.10g")
-    elapsed = round(time.time() - started, 3)
     print("RESULT_JSON " + json.dumps({
         "n_rows": int(len(output)),
-        "train_seconds": elapsed,
-        "notes": "LightGBM LambdaRank on per-user groups with official five categorical fields",
+        "train_seconds": round(time.time() - started, 3),
+        "notes": "LightGBM LambdaRank official five fields; increased L2 regularisation to 10",
     }))
 
 
```

**Resulting metrics**

| metric | value | vs baseline |
|---|---|---|
| gauc | 0.63974 | -0.02770 |
| ndcg@5 | 0.52405 | -0.01160 |
| primary | 0.58190 | -0.01970 |

### Iteration 11 — `n010` (improve from `n000`)

**Hypothesis** — what it intended to try, and why

> The five-field FM is presently optimized with pointwise BCE even though both validation metrics judge only within-user order; its 0.5715 primary indicates that calibrated global long-view probabilities are not transferring to the short held-out lists. Replacing only the training objective with BPR comparisons sampled by positive impressions (thus weighting users by positive count, as GAUC does) should make the same FM allocate its capacity to user-specific order and plausibly recover several points of primary score.

**Plan**

- Keep the data loading, five categorical fields, FM architecture, feature encoding, split discipline, user-group subsampling, and row-order-safe submission unchanged.
- Replace pointwise BCE training with a BPR pairwise loss between each positive training impression and a randomly sampled negative impression from the same user.
- Generate one pair per eligible positive per epoch; this naturally samples users in proportion to their positive count rather than uniformly.
- Continue to use the fixed 40-epoch schedule, so test fitting uses train+validation labels without any held-out-label selection.

**Ideas drawn on:** T1.pair-weighting-sweep

**Code diff applied**

```diff
--- nodes/n000/pipeline.py
+++ nodes/n010/pipeline.py
@@ -1,5 +1,4 @@
 import argparse
-import csv
 import json
 import os
 import random
@@ -27,7 +26,6 @@
 
 
 def read_log(path):
-    # IDs are strings so their representation is retained for submission and lookup.
     return pd.read_csv(path, dtype={"user_id": str, "video_id": str, "tab": str})
 
 
@@ -41,7 +39,6 @@
 
 
 def add_author(frame, author_lookup):
-    # Series.map is a one-to-one left lookup and therefore cannot reorder or duplicate rows.
     out = frame.copy()
     n = len(out)
     out["author_id"] = out["video_id"].map(author_lookup).fillna("__UNKNOWN_AUTHOR__").astype(str)
@@ -51,8 +48,6 @@
 
 def duration_bucket(values):
     x = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
-    # Log-second bins give a compact categorical representation while retaining the
-    # known non-linear duration relationship.
     return np.minimum(np.floor(np.log2(np.maximum(x, 0.0) / 1000.0 + 1.0)).astype(np.int32), 20).astype(str)
 
 
@@ -68,7 +63,6 @@
         a = tr[field].fillna("__MISSING__").astype(str)
         b = te[field].fillna("__MISSING__").astype(str)
         vocabulary = pd.Index(a.unique())
-        # Code zero is reserved for categories not available at fitting time.
         ca = vocabulary.get_indexer(a) + 1
         cb = vocabulary.get_indexer(b) + 1
         blocks_train.append((ca + offset).astype(np.int64))
@@ -103,7 +97,6 @@
     rng = np.random.default_rng(seed)
     keep_n = max(1, int(np.ceil(len(users) * fraction)))
     keep = set(rng.choice(users, size=keep_n, replace=False).tolist())
-    # This deliberately filters complete user groups, never individual impressions.
     return frame.loc[frame["user_id"].isin(keep)].copy()
 
 
@@ -113,26 +106,52 @@
     with torch.no_grad():
         for start in range(0, len(x), batch_size):
             z = torch.from_numpy(x[start:start + batch_size])
-            ans.append(torch.sigmoid(model(z)).cpu().numpy())
+            ans.append(model(z).cpu().numpy())
     return np.concatenate(ans) if ans else np.empty(0, dtype=np.float64)
 
 
-def train_fm(x, y, seed, epochs):
+def eligible_groups(users, y):
+    """Return positive and negative row-index arrays for users having both labels."""
+    groups = []
+    order = np.argsort(users, kind="stable")
+    sorted_users = users[order]
+    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
+    for idx in np.split(order, cuts):
+        pos = idx[y[idx] > 0.5]
+        neg = idx[y[idx] <= 0.5]
+        if len(pos) and len(neg):
+            groups.append((pos.astype(np.int64), neg.astype(np.int64)))
+    return groups
+
+
+def train_fm_bpr(x, y, users, seed, epochs):
     torch.manual_seed(seed)
     model = FactorizationMachine(int(x.max()) + 1, k=16)
     opt = torch.optim.Adam(model.parameters(), lr=0.001)
     tx = torch.from_numpy(x)
-    ty = torch.from_numpy(y.astype(np.float32))
+    groups = eligible_groups(users, y)
+    if not groups:
+        raise RuntimeError("no users with both positive and negative training impressions")
+    rng = np.random.default_rng(seed)
     batch = 8192
-    # Fixed epochs make the test fit use all train+validation labels without selecting
-    # against unavailable test labels. This is the published baseline schedule.
-    for epoch in range(epochs):
-        model.train()
-        order = torch.randperm(len(tx))
-        for start in range(0, len(tx), batch):
-            idx = order[start:start + batch]
-            logits = model(tx[idx])
-            loss = F.binary_cross_entropy_with_logits(logits, ty[idx])
+    for _epoch in range(epochs):
+        # Each eligible positive appears once. Consequently a user's pair mass is its
+        # positive count, rather than one equal vote per user.
+        pos_parts = []
+        neg_parts = []
+        for pos, neg in groups:
+            pos_parts.append(pos)
+            neg_parts.append(neg[rng.integers(0, len(neg), size=len(pos))])
+        pos_idx = np.concatenate(pos_parts)
+        neg_idx = np.concatenate(neg_parts)
+        perm = rng.permutation(len(pos_idx))
+        pos_idx = pos_idx[perm]
+        neg_idx = neg_idx[perm]
+        for start in range(0, len(pos_idx), batch):
+            p = torch.from_numpy(pos_idx[start:start + batch])
+            n = torch.from_numpy(neg_idx[start:start + batch])
+            margin = model(tx[p]) - model(tx[n])
+            loss = F.softplus(-margin).mean()
             opt.zero_grad(set_to_none=True)
             loss.backward()
             opt.step()
@@ -160,11 +179,14 @@
     author_lookup = video.set_index("video_id")["author_id"]
 
     fit_raw = choose_users(fit_raw, args.subsample, args.seed)
+    labels = pd.to_numeric(fit_raw["long_view"], errors="coerce")
+    # Missing labels cannot form a supervised comparison. This only affects fitting rows.
```
_diff truncated at 120 of 140 lines; full source at `nodes/n010/pipeline.py`_

**Resulting metrics**

| metric | value | vs baseline |
|---|---|---|
| gauc | 0.63635 | -0.03110 |
| ndcg@5 | 0.52254 | -0.01320 |
| primary | 0.57945 | -0.02220 |

### Iteration 12 — `n011` (improve from `n010`)

**Hypothesis** — what it intended to try, and why

> The positive-count-weighted BPR FM is likely spending too many of its pair updates on negatives that it already separates from a user's positives, while the short-list nDCG@5 component is determined disproportionately by the few negatives receiving high scores. Replacing only the per-user uniform negative draw with a draw from each user's current top-scoring negative quartile after a warm-up epoch should focus the same model capacity on ordering-confusing impressions and raise validation primary by roughly 0.002–0.005; if it does not, hard-negative concentration is too noisy for these small user lists.

**Plan**

- Keep the split loading, five FM fields, architecture, optimizer, epoch count, positive-count user weighting, and output ordering unchanged.
- Leave epoch 1's negative sampling uniform as a warm-up.
- Before every later epoch, score the fitting rows with the current FM and, for each eligible user, sample each positive's negative from that user's highest-scoring 25% of negative impressions.
- Continue to draw one pair per positive, so users retain pair mass proportional to their number of positive impressions.
- Use only fitting labels and fitting impressions for hard-negative selection; evaluation rows and labels are never inspected during training.

**Ideas drawn on:** T1.hard-negative-pairs

**Code diff applied**

```diff
--- nodes/n010/pipeline.py
+++ nodes/n011/pipeline.py
@@ -134,19 +134,27 @@
         raise RuntimeError("no users with both positive and negative training impressions")
     rng = np.random.default_rng(seed)
     batch = 8192
-    for _epoch in range(epochs):
-        # Each eligible positive appears once. Consequently a user's pair mass is its
-        # positive count, rather than one equal vote per user.
+    for epoch in range(epochs):
+        # The first pass is deliberately uniform. Later passes mine only negatives that
+        # the current model would place near the top of this user's own shown list.
+        fitted_scores = None if epoch == 0 else predict(model, x)
         pos_parts = []
         neg_parts = []
         for pos, neg in groups:
+            if fitted_scores is None:
+                candidates = neg
+            else:
+                take = max(1, int(np.ceil(0.25 * len(neg))))
+                top_at = np.argpartition(fitted_scores[neg], len(neg) - take)[-take:]
+                candidates = neg[top_at]
             pos_parts.append(pos)
-            neg_parts.append(neg[rng.integers(0, len(neg), size=len(pos))])
+            neg_parts.append(candidates[rng.integers(0, len(candidates), size=len(pos))])
         pos_idx = np.concatenate(pos_parts)
         neg_idx = np.concatenate(neg_parts)
         perm = rng.permutation(len(pos_idx))
         pos_idx = pos_idx[perm]
         neg_idx = neg_idx[perm]
+        model.train()
         for start in range(0, len(pos_idx), batch):
             p = torch.from_numpy(pos_idx[start:start + batch])
             n = torch.from_numpy(neg_idx[start:start + batch])
@@ -180,7 +188,6 @@
 
     fit_raw = choose_users(fit_raw, args.subsample, args.seed)
     labels = pd.to_numeric(fit_raw["long_view"], errors="coerce")
-    # Missing labels cannot form a supervised comparison. This only affects fitting rows.
     fit_raw = fit_raw.loc[labels.notna()].copy()
     y = labels.loc[labels.notna()].to_numpy(dtype=np.float32)
     fit = add_author(fit_raw, author_lookup)
@@ -198,7 +205,8 @@
     os.makedirs(args.out_dir, exist_ok=True)
     result.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False,
                   columns=["row_id", "user_id", "video_id", "score"], float_format="%.8f")
-    notes = "five-field k=16 FM with positive-count-weighted within-user BPR; train-only" if args.split == "val" else "five-field k=16 FM with positive-count-weighted within-user BPR; train+validation"
+    mode = "train-only" if args.split == "val" else "train+validation"
+    notes = "five-field k=16 FM with positive-count-weighted BPR and top-quartile within-user hard-negative sampling; " + mode
     print("RESULT_JSON " + json.dumps({"n_rows": int(len(result)), "train_seconds": round(time.time() - started, 3), "notes": notes}))
 
 
```

**Resulting metrics**

| metric | value | vs baseline |
|---|---|---|
| gauc | 0.62747 | -0.03990 |
| ndcg@5 | 0.52014 | -0.01560 |
| primary | 0.57380 | -0.02780 |

