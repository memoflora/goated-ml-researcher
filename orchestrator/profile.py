"""Automatic EDA — the capability that lets the agent face a dataset nobody has seen.

    profile(task) -> dict of facts

Until now the data card was hand-written: a human looked at KuaiRand, decided what mattered,
and typed it out. That does not scale to "give it any dataset", because the card is the
agent's *only* view of the data — no card, no informed hypothesis.

So this module derives the card's facts mechanically: split sizes, the target's shape,
per-column types and cardinalities and missingness, how each feature relates to the target,
and the handful of structural traps that are worth warning about (leakage-grade correlation,
constant columns, identifier-like columns, and columns that exist in train but not in the
split being predicted).

**Facts only.** Nothing here suggests a model or a feature. What to *try* is the idea bank's
job — keeping that boundary is what lets the journal attribute each attempt to an idea.

Deliberately capped: a 900-column dataset must not blow the prompt budget, so columns are
ranked by how much they look related to the target and only the top slice is described in
full. The rest are counted and named.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from orchestrator import datasource as ds
from orchestrator.taskspec import TaskConfig

#: Describe at most this many columns in full; the rest are summarised.
MAX_DETAILED_COLUMNS = 40
#: Distinct values above which a categorical is called high-cardinality.
HIGH_CARD = 1000
#: |correlation| at or above this is flagged as possible leakage, not as a feature.
LEAK_CORR = 0.995


def _is_numeric(s) -> bool:
    return s.dtype.kind in "ifub"


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN-safe, 0.0 when undefined."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    a, b = x[mask], y[mask]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _target_facts(target: np.ndarray, kind: str) -> dict[str, Any]:
    finite = target[np.isfinite(target)] if target.dtype.kind in "fc" else target
    out: dict[str, Any] = {"n_missing": int(len(target) - len(finite))}

    if kind in ("binary", "multiclass"):
        vals, counts = np.unique(finite, return_counts=True)
        total = counts.sum() or 1
        out["type"] = "categorical"
        out["classes"] = [
            {"value": _py(v), "count": int(c), "rate": round(float(c / total), 4)}
            for v, c in zip(vals[:20], counts[:20], strict=False)
        ]
        out["n_classes"] = int(len(vals))
        if len(vals) == 2:
            out["positive_rate"] = round(float(counts[-1] / total), 4)
        return out

    x = finite.astype(np.float64)
    out["type"] = "numeric"
    if len(x):
        out.update(
            mean=round(float(x.mean()), 4),
            std=round(float(x.std()), 4),
            min=round(float(x.min()), 4),
            max=round(float(x.max()), 4),
            quantiles={
                str(q): round(float(np.quantile(x, q)), 4) for q in (0.01, 0.25, 0.5, 0.75, 0.99)
            },
            skew=round(float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12)), 3),
            n_zero=int((x == 0).sum()),
            n_negative=int((x < 0).sum()),
        )
    return out


def _py(v):
    """numpy scalar -> plain python, so the facts dict is JSON-serialisable."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def _column_facts(df, task: TaskConfig, y: np.ndarray | None) -> list[dict[str, Any]]:
    d = task.data
    skip = {d.target, *d.drop_columns}
    cols = []
    n = len(df)

    for name in df.columns:
        if name in skip:
            continue
        s = df[name]
        nunique = int(s.nunique(dropna=True))
        miss = float(s.isna().mean())
        info: dict[str, Any] = {
            "name": name,
            "dtype": str(s.dtype),
            "n_unique": nunique,
            "missing_pct": round(100 * miss, 2),
            "role": _role(name, d),
        }

        if _is_numeric(s):
            x = s.to_numpy(dtype=np.float64, na_value=np.nan)
            finite = x[np.isfinite(x)]
            info["kind"] = "numeric"
            if len(finite):
                info["min"] = round(float(finite.min()), 4)
                info["median"] = round(float(np.median(finite)), 4)
                info["max"] = round(float(finite.max()), 4)
            if y is not None and y.dtype.kind in "if":
                info["corr_with_target"] = round(_safe_corr(x, y), 4)
        else:
            info["kind"] = "categorical"
            top = s.value_counts(dropna=True).head(4)
            info["top_values"] = [
                {"value": str(_py(v))[:30], "rate": round(float(c / max(n, 1)), 4)}
                for v, c in top.items()
            ]
            if y is not None and y.dtype.kind in "if" and nunique <= 200:
                # target mean per level tells the agent whether a category separates at all
                grouped = df.groupby(name, observed=True)[task.data.target]
                try:
                    spread = float(grouped.mean().std())
                    info["target_mean_spread"] = round(spread, 4)
                except (TypeError, ValueError):
                    pass

        info["constant"] = nunique <= 1
        info["identifier_like"] = nunique >= 0.98 * n and n > 50
        info["high_cardinality"] = nunique >= HIGH_CARD
        cols.append(info)
    return cols


def _role(name: str, d) -> str:
    if name == d.group:
        return "group"
    if name in d.id_columns:
        return "id"
    if d.split.date_column and name == d.split.date_column:
        return "date"
    return "feature"


def _rank_columns(cols: list[dict]) -> list[dict]:
    """Most-informative first: strongest relationship to the target, then lowest missingness."""

    def key(c: dict) -> tuple:
        rel = abs(c.get("corr_with_target", 0.0)) or c.get("target_mean_spread", 0.0) or 0.0
        role_rank = {"group": 0, "id": 1, "date": 2}.get(c["role"], 3)
        return (role_rank, -rel, c["missing_pct"])

    return sorted(cols, key=key)


def _warnings(cols: list[dict], frames: dict, task: TaskConfig) -> list[str]:
    out: list[str] = []
    for c in cols:
        if c.get("constant"):
            out.append(f"`{c['name']}` is constant — it carries no signal.")
        if abs(c.get("corr_with_target", 0.0)) >= LEAK_CORR:
            out.append(
                f"`{c['name']}` correlates {c['corr_with_target']:+.4f} with the target. "
                "That is leakage-grade: check it is not a copy or a post-outcome field."
            )
        if c.get("identifier_like") and c["role"] == "feature":
            out.append(
                f"`{c['name']}` is near-unique per row, so it behaves like an identifier "
                "rather than a feature."
            )

    train_cols = set(frames["train"].columns) if "train" in frames else set()
    for split_name, df in frames.items():
        if split_name == "train":
            continue
        missing = train_cols - set(df.columns) - {task.data.target}
        if missing:
            names = ", ".join(f"`{m}`" for m in sorted(missing)[:8])
            out.append(
                f"{len(missing)} column(s) exist in train but not in {split_name}: {names}. "
                "Any feature built from them cannot be computed at prediction time."
            )
    return out


def profile(task: TaskConfig, *, max_rows: int | None = 400_000) -> dict[str, Any]:
    """Derive every fact the data card quotes. Deterministic given the data on disk."""
    frames = ds.all_frames(task)
    if "train" not in frames:
        raise KeyError(f"task {task.name!r} has no train split; got {sorted(frames)}")

    d = task.data
    facts: dict[str, Any] = {
        "task": {
            "name": task.name,
            "kind": task.kind,
            "description": task.description,
            "target": d.target,
            "group": d.group,
            "primary_parts": list(task.primary_parts),
            "report_metrics": list(task.report_metrics),
            "submission_columns": list(task.submission_columns),
            "prediction_column": task.prediction_column,
        },
        "splits": {},
        "warnings": [],
    }

    for name, df in sorted(frames.items()):
        entry: dict[str, Any] = {"rows": int(len(df)), "columns": int(df.shape[1])}
        if d.target in df.columns:
            tgt = df[d.target].to_numpy()
            entry["target"] = _target_facts(tgt, task.kind)
        else:
            entry["target"] = {"type": "absent", "note": "labels not present in this split"}
        if d.group and d.group in df.columns:
            g = df[d.group].to_numpy()
            _, counts = np.unique(g, return_counts=True)
            entry["groups"] = {
                "n_groups": int(len(counts)),
                "rows_per_group_median": int(np.median(counts)),
                "rows_per_group_mean": round(float(counts.mean()), 1),
                "rows_per_group_max": int(counts.max()),
            }
            if d.target in df.columns and task.kind in ("binary", "ranking"):
                tgt = df[d.target].to_numpy()
                order = np.argsort(g, kind="mergesort")
                gs, ts = g[order], tgt[order]
                edges = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
                sums = np.add.reduceat(ts, edges[:-1])
                sizes = np.diff(edges)
                entry["groups"]["zero_positive_pct"] = round(100 * float((sums == 0).mean()), 1)
                entry["groups"]["all_positive_pct"] = round(
                    100 * float((sums == sizes).mean()), 1
                )
        facts["splits"][name] = entry

    train = frames["train"]
    if max_rows and len(train) > max_rows:  # profiling is a one-off; still, be kind to memory
        train = train.sample(n=max_rows, random_state=0).reset_index(drop=True)
    y = None
    if d.target in train.columns:
        col = train[d.target]
        if _is_numeric(col):
            y = col.to_numpy(dtype=np.float64, na_value=np.nan)

    cols = _column_facts(train, task, y)
    facts["n_columns_total"] = len(cols)
    ranked = _rank_columns(cols)
    facts["columns"] = ranked[:MAX_DETAILED_COLUMNS]
    facts["columns_omitted"] = [c["name"] for c in ranked[MAX_DETAILED_COLUMNS:]]
    facts["warnings"] = _warnings(cols, frames, task)
    return facts
