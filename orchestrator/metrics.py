"""Metric registry — the part of scoring that is not tied to any one dataset.

    compute(name, y_true, y_pred, groups=None) -> float
    primary_of(metrics, parts) -> float

Every metric here is registered with a **direction**. The orchestrator's search, its
convergence rule and its `best` tracking all assume *higher is better*, so a metric where
lower is better (RMSE, log loss) is registered with `greater_is_better=False` and the
composite `primary` negates it. That way `core.py` never needs to know which is which —
`primary` is always something to maximise.

Implemented in numpy on purpose. The orchestrator must stay installable without
scikit-learn; the *generated pipeline* may use whatever `requirements-pipeline.txt` allows,
but the thing that grades it should have as few moving parts as possible.

KuaiRand-Pure is the exception: its `gauc` and `ndcg@5` delegate to the vendored starter
kit, which is the organisers' own code and the sole authority on their conventions. The
generic implementations here are for *other* ranking tasks, and `tests/test_metrics.py`
asserts the two agree on KuaiRand data so the generic path is trustworthy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EPS = 1e-15


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable[..., float]
    greater_is_better: bool
    needs_groups: bool = False
    #: Human-readable, goes into the data card and the prompt so the agent knows what it
    #: is being graded on without us hardcoding a description anywhere else.
    description: str = ""


_REGISTRY: dict[str, Metric] = {}


def register(
    name: str,
    *,
    greater_is_better: bool,
    needs_groups: bool = False,
    description: str = "",
) -> Callable:
    def deco(fn: Callable[..., float]) -> Callable[..., float]:
        _REGISTRY[name] = Metric(name, fn, greater_is_better, needs_groups, description)
        return fn

    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str) -> Metric:
    key = name.strip().lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    # ndcg@K / map@K / recall@K for any K
    for prefix in ("ndcg@", "map@", "recall@", "precision@"):
        if key.startswith(prefix) and key[len(prefix) :].isdigit():
            base = _REGISTRY[prefix + "k"]
            k = int(key[len(prefix) :])
            return Metric(
                key,
                lambda y, p, groups=None, _k=k, _f=base.fn: _f(y, p, groups=groups, k=_k),
                base.greater_is_better,
                base.needs_groups,
                base.description.replace("@K", f"@{k}"),
            )
    raise KeyError(f"unknown metric {name!r}; known: {', '.join(available())}, plus ndcg@K/map@K")


def compute(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
) -> float:
    m = get(name)
    if m.needs_groups and groups is None:
        raise ValueError(f"metric {name!r} needs a group column (e.g. user_id)")
    return float(m.fn(np.asarray(y_true), np.asarray(y_pred), groups=groups))


def primary_of(metrics: dict[str, float], parts: tuple[str, ...] | list[str]) -> float:
    """Composite score, always oriented so that **higher is better**.

    Equal-weighted mean of `parts`, with any lower-is-better member negated first.
    """
    if not parts:
        raise ValueError("primary needs at least one metric")
    total = 0.0
    for p in parts:
        v = float(metrics[p])
        total += v if get(p).greater_is_better else -v
    return total / len(parts)


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------


@register("rmse", greater_is_better=False, description="root mean squared error")
def _rmse(y, p, groups=None) -> float:
    return float(np.sqrt(np.mean((y.astype(float) - p.astype(float)) ** 2)))


@register("mae", greater_is_better=False, description="mean absolute error")
def _mae(y, p, groups=None) -> float:
    return float(np.mean(np.abs(y.astype(float) - p.astype(float))))


@register("rmsle", greater_is_better=False, description="root mean squared log error")
def _rmsle(y, p, groups=None) -> float:
    y = np.clip(y.astype(float), 0, None)
    p = np.clip(p.astype(float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))


@register("r2", greater_is_better=True, description="coefficient of determination")
def _r2(y, p, groups=None) -> float:
    y = y.astype(float)
    ss_res = float(np.sum((y - p.astype(float)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# binary classification
# ---------------------------------------------------------------------------


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via rank statistics, ties averaged. Undefined for a single class."""
    y = y.astype(np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sp = p[order]
    i = 0
    while i < len(sp):  # average ranks within tie groups
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@register("auc", greater_is_better=True, description="area under the ROC curve")
def _auc_metric(y, p, groups=None) -> float:
    return _auc(y, p)


@register("logloss", greater_is_better=False, description="binary cross-entropy")
def _logloss(y, p, groups=None) -> float:
    p = np.clip(p.astype(float), EPS, 1 - EPS)
    y = y.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@register("accuracy", greater_is_better=True, description="fraction correct")
def _accuracy(y, p, groups=None) -> float:
    pred = p if p.dtype.kind in "iu" else (p.astype(float) >= 0.5).astype(np.int64)
    return float(np.mean(pred == y))


@register("f1", greater_is_better=True, description="F1 of the positive class")
def _f1(y, p, groups=None) -> float:
    pred = p if p.dtype.kind in "iu" else (p.astype(float) >= 0.5).astype(np.int64)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


@register("average_precision", greater_is_better=True, description="area under PR curve")
def _ap(y, p, groups=None) -> float:
    order = np.argsort(-p.astype(float), kind="mergesort")
    y = y.astype(np.int64)[order]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / n_pos)


# ---------------------------------------------------------------------------
# multiclass
# ---------------------------------------------------------------------------


@register("macro_f1", greater_is_better=True, description="unweighted mean per-class F1")
def _macro_f1(y, p, groups=None) -> float:
    classes = np.unique(y)
    scores = []
    for c in classes:
        tp = float(((p == c) & (y == c)).sum())
        fp = float(((p == c) & (y != c)).sum())
        fn = float(((p != c) & (y == c)).sum())
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# grouped ranking — the KuaiRand family, and any other within-group ranking task
# ---------------------------------------------------------------------------


def _group_slices(groups: np.ndarray) -> list[np.ndarray]:
    """Index arrays per group. Stable, so ties keep submission order."""
    order = np.argsort(groups, kind="mergesort")
    g = groups[order]
    edges = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
    return [order[edges[i] : edges[i + 1]] for i in range(len(edges) - 1)]


@register(
    "gauc",
    greater_is_better=True,
    needs_groups=True,
    description=(
        "group-wise AUC: per-group AUC averaged with weight = that group's positive count. "
        "Groups that are all-positive or all-negative are excluded."
    ),
)
def _gauc(y, p, groups=None) -> float:
    y = y.astype(np.int64)
    total_w = 0.0
    total = 0.0
    for idx in _group_slices(np.asarray(groups)):
        yy = y[idx]
        n_pos = int(yy.sum())
        if n_pos == 0 or n_pos == len(yy):
            continue
        a = _auc(yy, p[idx])
        if not np.isnan(a):
            total += a * n_pos
            total_w += n_pos
    return float(total / total_w) if total_w else float("nan")


@register(
    "ndcg@k",
    greater_is_better=True,
    needs_groups=True,
    description=(
        "normalised discounted cumulative gain at K, gain = 2^rel - 1. Groups with no "
        "positive score 0 and are **included** in the average."
    ),
)
def _ndcg_at_k(y, p, groups=None, k: int = 5) -> float:
    y = y.astype(np.float64)
    out = []
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    for idx in _group_slices(np.asarray(groups)):
        yy, pp = y[idx], p[idx]
        top = np.argsort(-pp, kind="mergesort")[:k]
        gains = (2 ** yy[top] - 1) * discount[: len(top)]
        ideal_rel = np.sort(yy)[::-1][:k]
        ideal = (2**ideal_rel - 1) * discount[: len(ideal_rel)]
        denom = ideal.sum()
        out.append(float(gains.sum() / denom) if denom > 0 else 0.0)
    return float(np.mean(out)) if out else 0.0


@register(
    "map@k",
    greater_is_better=True,
    needs_groups=True,
    description="mean average precision at K over groups",
)
def _map_at_k(y, p, groups=None, k: int = 5) -> float:
    y = y.astype(np.int64)
    out = []
    for idx in _group_slices(np.asarray(groups)):
        yy, pp = y[idx], p[idx]
        top = np.argsort(-pp, kind="mergesort")[:k]
        rel = yy[top]
        if rel.sum() == 0:
            out.append(0.0)
            continue
        hits = np.cumsum(rel)
        prec = hits / np.arange(1, len(rel) + 1)
        out.append(float((prec * rel).sum() / min(int(yy.sum()), k)))
    return float(np.mean(out)) if out else 0.0


@register(
    "recall@k",
    greater_is_better=True,
    needs_groups=True,
    description="fraction of a group's positives that appear in the top K",
)
def _recall_at_k(y, p, groups=None, k: int = 5) -> float:
    y = y.astype(np.int64)
    out = []
    for idx in _group_slices(np.asarray(groups)):
        yy, pp = y[idx], p[idx]
        n_pos = int(yy.sum())
        if n_pos == 0:
            out.append(0.0)
            continue
        top = np.argsort(-pp, kind="mergesort")[:k]
        out.append(float(yy[top].sum() / n_pos))
    return float(np.mean(out)) if out else 0.0
