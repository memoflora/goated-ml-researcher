"""Reference pipeline: the official FM, trained with a within-user ranking loss.

NOT A SUBMISSION. This is calibration. Its job is to answer one question before the agent
spends fifty iterations on it:

    Is there real headroom above the baseline, and is the objective mismatch where it is?

The organisers rank "change the loss function" as the most likely untested win on this
dataset, and our entire T1 tier is built on that claim. If it is wrong, we want to know at
H+11, not at H+40 with the run already committed.

This is a controlled experiment, so exactly one thing changes:

    same 5 fields, same encoder      (vendor/starter_kit/data.py, untouched)
    same FM scorer, same k=16        (copied from baseline.py, not reimplemented)
    same Adam, same lr, same l2
    same early stopping on validation primary
    ------------------------------------------------------------------
    pointwise logloss  ->  within-user pairwise BPR

Any delta is therefore attributable to the loss alone. That is the whole point; a "better
pipeline" that changed four things would prove nothing about which one mattered.

Trains on train, selects on validation. Never reads test - see reference/README.md.

    python reference/bpr_fm.py --epochs 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor" / "starter_kit"))

from data import FIELDS, encode, load  # noqa: E402  (vendored, path-injected above)
from evaluate import evaluate  # noqa: E402

DEFAULT_DATA = REPO / "data" / "KuaiRand-Pure" / "data"
BASELINE_VAL = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
SEED_STD = 0.0008


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Identical to baseline.py's FM. Copied rather than imported so the two cannot
    drift, and so the only difference between the runs is visibly the loss."""

    def __init__(self, dim: int, k: int = 16, lr: float = 0.001, l2: float = 1e-6, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def _parts(self, X: np.ndarray):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.W[X].sum(1) + inter, E, S

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._parts(X)[0]

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        return np.concatenate([self.score(X[i:i + bs]) for i in range(0, len(X), bs)])

    def _adam(self, gV: np.ndarray, gW: np.ndarray) -> None:
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

    def bpr_step(self, Xp: np.ndarray, Xn: np.ndarray) -> float:
        """One BPR update over (positive, negative) pairs drawn from the same user.

        loss = -log sigmoid(s_p - s_n).  With c = sigmoid(-(s_p - s_n)):
            dL/ds_p = -c      dL/ds_n = +c

        Note what is missing: there is no global bias term. In a pairwise difference any
        quantity constant within the user cancels exactly - the bias, and equally any
        pure user-side first-order feature. That is the same structural fact the
        organisers measured (item_pop x user_bias scores identically to item_pop), here
        as an algebraic identity rather than an experiment.
        """
        B = len(Xp)
        sp, Ep, Sp = self._parts(Xp)
        sn, En, Sn = self._parts(Xn)
        d = sp - sn
        c = (sigmoid(-d) / B).astype(np.float32)        # (B,)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, Xp, -c[:, None])
        np.add.at(gW, Xn, c[:, None])
        np.add.at(gV, Xp, -c[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, c[:, None, None] * (Sn[:, None, :] - En))
        self._adam(gV, gW)
        return float(-np.mean(np.log(sigmoid(d) + 1e-9)))


def build_pair_pools(users: list[str], y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Group train rows by user into positive and negative index arrays.

    Only users with at least one of each can contribute a pair - which is also exactly
    the set GAUC scores, so the training signal and the metric look at the same users.
    """
    codes, _ = _factorise(users)
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    bounds = np.flatnonzero(np.diff(codes_sorted)) + 1
    pos_pools: list[np.ndarray] = []
    neg_pools: list[np.ndarray] = []
    for chunk in np.split(order, bounds):
        labels = y[chunk]
        p = chunk[labels == 1]
        n = chunk[labels == 0]
        if len(p) and len(n):
            pos_pools.append(p)
            neg_pools.append(n)
    return pos_pools, neg_pools


def _factorise(values: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    lookup: dict[str, int] = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        code = lookup.get(v)
        if code is None:
            code = len(lookup)
            lookup[v] = code
        out[i] = code
    return out, lookup


def sample_pairs(
    pos_pools: list[np.ndarray],
    neg_pools: list[np.ndarray],
    n_pairs: int,
    rng: np.random.Generator,
    weighting: str = "positives",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (positive, negative) pairs from within the same user.

    `weighting` decides how much each user contributes:

    - "uniform"   - every user equally often, regardless of how much data they have.
    - "positives" - each user in proportion to their positive count. This is what GAUC
      itself does: it averages per-user AUC **weighted by positive count**. Uniform
      sampling optimises a different quantity from the one we are scored on, so this is
      the default.

    The distinction is not cosmetic. Users have very unequal positive counts here, so the
    two objectives put their capacity in quite different places.
    """
    if weighting == "uniform":
        u = rng.integers(0, len(pos_pools), size=n_pairs)
    else:
        # Drawing a positive row uniformly from all positives is exactly the same thing
        # as drawing a user in proportion to their positive count.
        weights = np.array([len(p) for p in pos_pools], dtype=np.float64)
        u = rng.choice(len(pos_pools), size=n_pairs, p=weights / weights.sum())

    pos = np.empty(n_pairs, dtype=np.int64)
    neg = np.empty(n_pairs, dtype=np.int64)
    for i, ui in enumerate(u):
        p, n = pos_pools[ui], neg_pools[ui]
        pos[i] = p[rng.integers(len(p))]
        neg[i] = n[rng.integers(len(n))]
    return pos, neg


def run(data_dir: Path, *, k: int, lr: float, l2: float, epochs: int, bs: int,
        patience: int, pairs_per_epoch: int, seed: int, weighting: str = "positives") -> dict:
    print(f"loading {data_dir} ...")
    splits = load(str(data_dir))
    print({s: len(r) for s, r in splits.items()}, f"fields={FIELDS}")

    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]

    pos_pools, neg_pools = build_pair_pools(utr, ytr)
    print(f"pairable users: {len(pos_pools):,} of {len(set(utr)):,}")

    m = FM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    history: list[float] = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos, neg = sample_pairs(pos_pools, neg_pools, pairs_per_epoch, rng, weighting)
        losses = [
            m.bpr_step(Xtr[pos[i:i + bs]], Xtr[neg[i:i + bs]])
            for i in range(0, pairs_per_epoch, bs)
        ]
        va = evaluate(uva, yva, m.predict(Xva))
        history.append(va["primary"])
        delta = va["primary"] - BASELINE_VAL["primary"]
        print(
            f"  epoch {ep:2d} | bpr {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
            f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} "
            f"({delta:+.4f} vs baseline) | {time.time() - t0:.1f}s"
        )
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy())
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}")
                break

    assert best_state is not None
    m.V, m.W = best_state
    return {"valid": evaluate(uva, yva, m.predict(Xva)), "history": history}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--pairs-per-epoch", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weighting", choices=["positives", "uniform"], default="positives",
                    help="how much each user contributes to the pair stream")
    a = ap.parse_args()

    res = run(
        Path(a.data_dir), k=a.k, lr=a.lr, l2=a.l2, epochs=a.epochs, bs=a.bs,
        patience=a.patience, pairs_per_epoch=a.pairs_per_epoch, seed=a.seed,
        weighting=a.weighting,
    )
    v = res["valid"]
    print(f"\n=== BPR-FM (seed={a.seed}) — validation ===")
    print(f"{'':14}{'GAUC':>9}{'nDCG@5':>9}{'primary':>10}")
    print(f"{'baseline':14}{BASELINE_VAL['GAUC']:>9.4f}{BASELINE_VAL['nDCG@5']:>9.4f}"
          f"{BASELINE_VAL['primary']:>10.4f}")
    print(f"{'BPR-FM':14}{v['GAUC']:>9.4f}{v['nDCG@5']:>9.4f}{v['primary']:>10.4f}")
    print(f"{'delta':14}{v['GAUC'] - BASELINE_VAL['GAUC']:>+9.4f}"
          f"{v['nDCG@5'] - BASELINE_VAL['nDCG@5']:>+9.4f}"
          f"{v['primary'] - BASELINE_VAL['primary']:>+10.4f}")
    d = v["primary"] - BASELINE_VAL["primary"]
    verdict = (
        "REAL - beyond seed noise" if d > 2 * SEED_STD
        else "INSIDE NOISE - not evidence" if d > -2 * SEED_STD
        else "WORSE than baseline"
    )
    print(f"\n{d:+.4f} primary vs baseline: {verdict} (2 sigma = {2 * SEED_STD:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
