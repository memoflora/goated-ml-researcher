"""Factorization Machine on KuaiRand-Pure — the organisers' official baseline, re-expressed
against the frozen pipeline contract.

    python pipeline.py --data-dir DIR --out-dir DIR --split {val,test} --seed N [--subsample F]

Companion to `kuairand_pop.py`. Where that one proves the seam with a pure statistic, this
one proves it with a *trained* model: encoding, minibatch Adam, early stopping, and a
refit — the shape of everything the agent will actually write.

Faithfulness
------------
The encoder, the model, the optimiser and the training loop are the organisers'
`vendor/starter_kit/data.py::encode` and `baseline.py::run_fm`, reproduced here rather than
imported because the pipeline contract requires one self-contained file. The reproduction
is exact by construction:

  * the vocabulary is built in first-appearance order over the fit rows (`pd.factorize`
    assigns the same ids the original `dict` loop does), with the UNK slot last;
  * duration buckets use the same 9 quantile edges and the same `searchsorted`;
  * `FM`, its Adam state and its update are copied line for line;
  * the epoch loop draws from `np.random.default_rng(seed)` in the same order, uses the
    same batch size, the same `> best + 1e-5` improvement test and the same patience.

`tests/test_evaluator.py` asserts the result: validation primary 0.6016, the published
baseline, reached through `sandbox.run -> evaluate.validate -> evaluate.score`.

Why there is a metric in this file
----------------------------------
Early stopping needs a validation signal *inside* the run, exactly as the official baseline
has one. That is model selection, not scoring: the submission it writes is still graded
outside, by `orchestrator/evaluate.py`, which delegates to `vendor/starter_kit/`. The
vendored kit remains the sole authority — the copy below exists so that this fixture picks
the same epoch the published baseline picked, and for nothing else.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time

import numpy as np
import pandas as pd

# --- dataset facts, from the organisers' data.py ----------------------------------------
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
VIDEO_FEATURES = "video_features_basic_pure.csv"
LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
SPLIT_ALIASES = {"val": "valid", "valid": "valid", "test": "test", "train": "train"}
FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")

# baseline.py --model fm defaults
K = 16
LR = 0.001
L2 = 1e-6
EPOCHS = 40
BATCH = 8192
PATIENCE = 4
N_BUCKETS = 10


# --------------------------------------------------------------------------- metric copy
# vendor/starter_kit/evaluate.py, verbatim. Used ONLY to choose the stopping epoch.
def auc(labels, scores):
    pairs = sorted(zip(scores, labels, strict=True))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for kk in range(i, j + 1):
            ranks[kk] = avg
        i = j + 1
    npos = sum(lab for _, lab in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    srank = sum(r for r, (_, lab) in zip(ranks, pairs, strict=True) if lab == 1)
    return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)


def ndcg_at_k(labels, k):
    disc = [math.log2(i + 2) for i in range(k)]
    dcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(labels[:k]))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate(user_ids, labels, scores, k=5):
    byu = collections.defaultdict(list)
    for u, y, s in zip(user_ids, labels, scores, strict=True):
        byu[u].append((s, y))
    gnum = gden = 0.0
    nd = []
    for lst in byu.values():
        lst.sort(key=lambda x: -x[0])
        labs = [y for _, y in lst]
        npos = sum(labs)
        if 0 < npos < len(labs):
            gnum += npos * auc(labs, [s for s, _ in lst])
            gden += npos
        nd.append(ndcg_at_k(labs, k))
    gauc = gnum / gden if gden else 0.5
    ndcg = sum(nd) / len(nd) if nd else 0.0
    return {"GAUC": gauc, f"nDCG@{k}": ndcg, "primary": (gauc + ndcg) / 2.0}


# --------------------------------------------------------------------------- data
def resolve_data_dir(data_dir: str) -> str:
    for c in (data_dir, os.path.join(data_dir, "KuaiRand-Pure", "data"),
              os.path.join(data_dir, "data")):
        if os.path.isfile(os.path.join(c, LOG_FILES[0])):
            return c
    raise FileNotFoundError(f"could not find {LOG_FILES[0]} under {data_dir!r}")


def load_logs(data_dir: str) -> pd.DataFrame:
    """The two standard logs in file order, joined to the video-side author id.

    Nothing here sorts or dedupes: the concatenated file order *is* the row_id order.
    """
    vid2author = pd.read_csv(
        os.path.join(data_dir, VIDEO_FEATURES),
        usecols=["video_id", "author_id"], encoding="utf-8",
    ).set_index("video_id")["author_id"]

    frames = [
        pd.read_csv(
            os.path.join(data_dir, f),
            usecols=["user_id", "video_id", "date", "tab", "duration_ms", LABEL],
            encoding="utf-8",
        )
        for f in LOG_FILES
    ]
    df = pd.concat(frames, ignore_index=True)
    # data.py: vid2author.get(video_id, 'UNK'). Unknown authors share one vocabulary slot.
    df["author_id"] = df["video_id"].map(vid2author).astype("object")
    df.loc[df["author_id"].isna(), "author_id"] = "UNK"
    df["y"] = (df[LABEL].to_numpy() != 0).astype(np.float32)
    return df


def slice_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    lo, hi = SPLITS[name]
    date = df["date"].to_numpy()
    return df.loc[(date >= lo) & (date <= hi)]


def subsample_users(df: pd.DataFrame, frac, seed: int) -> pd.DataFrame:
    """Whole users, never individual rows — the metric is computed within a user."""
    if frac is None or float(frac) >= 1.0:
        return df
    users = np.unique(df["user_id"].to_numpy())
    rng = np.random.default_rng(seed)
    keep = rng.choice(users, size=max(1, int(round(len(users) * float(frac)))),
                      replace=False)
    return df.loc[np.isin(df["user_id"].to_numpy(), keep)]


def bucket_edges(durations: np.ndarray) -> np.ndarray:
    """data.py::_bucket_edges — the 9 interior deciles of the fit set's duration_ms."""
    return np.quantile(np.asarray(durations, dtype=np.float64),
                       np.linspace(0, 1, N_BUCKETS + 1)[1:-1])


def encode(fit: pd.DataFrame, others: dict[str, pd.DataFrame]):
    """data.py::encode, vectorised. Vocabularies come from `fit` in first-appearance order;
    every value unseen there lands in that field's UNK slot."""
    edges = bucket_edges(fit["duration_ms"].to_numpy())

    def raw_cols(d: pd.DataFrame) -> list[np.ndarray]:
        return [
            d["user_id"].to_numpy(),
            d["video_id"].to_numpy(),
            d["author_id"].to_numpy(),
            d["tab"].to_numpy(),
            np.searchsorted(edges, d["duration_ms"].to_numpy(dtype=np.float64)),
        ]

    fit_cols = raw_cols(fit)
    vocabs, unk, dims = [], [], []
    for col in fit_cols:
        # factorize assigns 0,1,2,... in order of first appearance — the same ids the
        # original `if v not in vocab: vocab[v] = len(vocab)` loop assigns.
        _, uniques = pd.factorize(col, sort=False)
        vocabs.append(pd.Index(uniques))
        unk.append(len(uniques))          # the UNK slot sits one past the last real id
        dims.append(len(uniques) + 1)
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)

    def to_X(d: pd.DataFrame) -> np.ndarray:
        cols = raw_cols(d)
        X = np.empty((len(d), len(FIELDS)), dtype=np.int32)
        for i, col in enumerate(cols):
            idx = vocabs[i].get_indexer(col)
            X[:, i] = np.where(idx < 0, unk[i], idx) + offsets[i]
        return X

    out = {"__fit__": (to_X(fit), fit["y"].to_numpy(dtype=np.float32))}
    for name, d in others.items():
        out[name] = (to_X(d), d["y"].to_numpy(dtype=np.float32))
    return out, int(sum(dims))


# --------------------------------------------------------------------------- model
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """baseline.py::FM, copied. k=16, Adam, l2 on the embeddings and the linear term."""

    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def train_with_early_stopping(m, Xtr, ytr, Xva, yva, uva, seed, epochs=EPOCHS,
                              patience=PATIENCE):
    """baseline.py::run_fm's loop. Returns (best_state, best_primary, best_epoch)."""
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), BATCH):
            m.step(Xtr[idx[i:i + BATCH]], ytr[idx[i:i + BATCH]])
        p = evaluate(uva, yva.astype(int).tolist(), m.predict(Xva).tolist())["primary"]
        if p > best + 1e-5:
            best, bad, best_epoch = p, 0, ep
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state, best, best_epoch


def train_fixed_epochs(m, Xtr, ytr, seed, epochs):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), BATCH):
            m.step(Xtr[idx[i:i + BATCH]], ytr[idx[i:i + BATCH]])
    return m


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="KuaiRand-Pure FM baseline")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", required=True, choices=["val", "valid", "test"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--subsample", type=float, default=None)
    return p.parse_args(argv)


def write_submission(path, users, videos, scores):
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
    valid = slice_split(df, "valid")
    # --subsample never touches the evaluation rows: it thins the *fit* set, by whole
    # users, wherever one is formed below.
    train = subsample_users(slice_split(df, "train"), a.subsample, a.seed)

    if eval_split == "valid":
        # Contract §1.2: `--split val` fits on train only, and validation is what we
        # early-stop against.
        enc, dim = encode(train, {"eval": valid})
        Xtr, ytr = enc["__fit__"]
        Xva, yva = enc["eval"]
        m = FM(dim, seed=a.seed)
        state, best, best_epoch = train_with_early_stopping(
            m, Xtr, ytr, Xva, yva, valid["user_id"].to_numpy().tolist(), a.seed
        )
        m.V, m.W, m.b = state
        ev, preds = valid, m.predict(Xva)
        notes = (f"FM k={K} lr={LR}, fit on train, early stop at epoch {best_epoch} "
                 f"(internal valid primary {best:.4f}), dim={dim}")
    else:
        # `--split test` fits on train + validation. There is then no held-out week left
        # to stop on, so the epoch count is the one validation chose on train alone, and
        # the model is refit from scratch on the union for exactly that many epochs.
        enc0, dim0 = encode(train, {"eval": valid})
        Xtr0, ytr0 = enc0["__fit__"]
        Xva0, yva0 = enc0["eval"]
        _, best, best_epoch = train_with_early_stopping(
            FM(dim0, seed=a.seed), Xtr0, ytr0, Xva0, yva0,
            valid["user_id"].to_numpy().tolist(), a.seed,
        )
        fit = subsample_users(
            pd.concat([slice_split(df, "train"), valid], ignore_index=True),
            a.subsample, a.seed,
        )
        test = slice_split(df, "test")
        enc, dim = encode(fit, {"eval": test})
        Xfit, yfit = enc["__fit__"]
        Xte, _ = enc["eval"]
        m = train_fixed_epochs(FM(dim, seed=a.seed), Xfit, yfit, a.seed, best_epoch)
        ev, preds = test, m.predict(Xte)
        notes = (f"FM k={K} lr={LR}, {best_epoch} epochs chosen on validation "
                 f"(primary {best:.4f}), refit on train+valid, dim={dim}")

    train_seconds = time.time() - started
    os.makedirs(a.out_dir, exist_ok=True)
    write_submission(
        os.path.join(a.out_dir, "submission.csv"),
        ev["user_id"].to_numpy(), ev["video_id"].to_numpy(),
        np.asarray(preds, dtype=np.float64),
    )
    print("RESULT_JSON " + json.dumps({
        "n_rows": int(len(preds)),
        "train_seconds": round(float(train_seconds), 3),
        "notes": notes,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
