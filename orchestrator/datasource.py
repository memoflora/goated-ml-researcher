"""Generic tabular loading and splitting, for any task described by a `TaskConfig`.

    eval_arrays(task, "valid") -> EvalArrays   # ids + target + groups, cached, fast
    frame(task, "train")       -> DataFrame    # the whole table, for profiling only

The orchestrator itself needs very little of a dataset: to score a submission it needs the
target, the group column if the metric is grouped, and the id columns to check alignment.
It never needs the features — the *generated pipeline* reads those itself. So the hot path
caches only those few arrays as an `.npz`, exactly like the KuaiRand loader does, and the
full frame is materialised only when the data card is built (once per dataset).

Row order is the contract. `row_id` is the 0-based position of a row inside its split, and
every split here is produced deterministically:

- `predefined` — one file per split, original file order preserved
- `date`       — filter one or more files by an inclusive range on a date column
- `random`     — a seeded permutation, then contiguous slices
- `group`      — whole groups held out, never individual rows, so that grouped metrics
                 (GAUC, nDCG) stay meaningful. Row sampling would silently break them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from orchestrator.taskspec import TaskConfig

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

_SPLIT_ALIASES = {"val": "valid", "valid": "valid", "train": "train", "test": "test"}


def normalise_split(split: str) -> str:
    key = str(split).strip().lower()
    if key not in _SPLIT_ALIASES:
        raise ValueError(f"unknown split {split!r}; expected train/val/valid/test")
    return _SPLIT_ALIASES[key]


@dataclass(frozen=True)
class EvalArrays:
    """Everything needed to score one split, in `row_id` order."""

    name: str
    target: np.ndarray
    groups: np.ndarray | None
    ids: dict[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.target.shape[0])


def _pandas():
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        raise RuntimeError(
            "generic tabular tasks need pandas; add it to requirements.txt"
        ) from None
    return pd


def _read_any(path: Path):
    """Read csv / tsv / parquet / json-lines by extension."""
    pd = _pandas()
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _source_files(task: TaskConfig) -> list[Path]:
    d = task.data
    if d.files:
        return [d.dir / v for v in d.files.values()]
    if d.file:
        return [d.dir / d.file]
    return []


def fingerprint(task: TaskConfig) -> str:
    """Identity of the underlying files plus the split plan, so caches self-invalidate."""
    h = hashlib.sha256()
    h.update(task.name.encode())
    h.update(repr(task.data.split).encode())
    h.update(repr(sorted(task.data.files.items())).encode())
    h.update(f"{task.data.file}|{task.data.target}|{task.data.group}".encode())
    for p in sorted(_source_files(task)):
        if p.exists():
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
        else:
            h.update(f"{p.name}:missing".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


def _split_frames(task: TaskConfig) -> dict[str, pd.DataFrame]:
    """Materialise every split as a DataFrame, in row_id order."""
    pd = _pandas()
    d = task.data
    plan = d.split

    if not d.dir.is_dir():
        raise FileNotFoundError(
            f"data directory not found: {d.dir}\n"
            f"Task {task.name!r} expects it. Create it, or fix `data.dir` in "
            f"{task.source_path or 'the task file'}."
        )

    if plan.strategy == "predefined":
        out = {}
        for split_name, fname in d.files.items():
            path = d.dir / fname
            if not path.is_file():
                raise FileNotFoundError(f"split {split_name!r}: no file at {path}")
            out[normalise_split(split_name)] = _read_any(path).reset_index(drop=True)
        return out

    # every other strategy starts from one concatenated table, file order preserved
    frames = [_read_any(p) for p in _source_files(task)]
    if not frames:
        raise FileNotFoundError(f"task {task.name!r}: no data files configured")
    full = pd.concat(frames, ignore_index=True)

    if plan.strategy == "date":
        col = plan.date_column
        if col not in full.columns:
            raise KeyError(f"date_column {col!r} is not in the data; columns: {list(full.columns)}")
        out = {}
        for split_name, (lo, hi) in plan.ranges.items():
            mask = (full[col] >= lo) & (full[col] <= hi)
            out[normalise_split(split_name)] = full[mask].reset_index(drop=True)
        return out

    if plan.strategy == "group":
        gcol = plan.group_column or d.group
        if not gcol or gcol not in full.columns:
            raise KeyError(f"group split needs a group column; got {gcol!r}")
        groups = full[gcol].to_numpy()
        uniq = np.unique(groups)
        rng = np.random.default_rng(plan.seed)
        rng.shuffle(uniq)
        n_valid = max(1, int(round(len(uniq) * plan.valid_frac)))
        n_test = int(round(len(uniq) * plan.test_frac))
        valid_g = set(uniq[:n_valid].tolist())
        test_g = set(uniq[n_valid : n_valid + n_test].tolist())
        train_g = set(uniq[n_valid + n_test :].tolist())
        out = {
            "train": full[np.isin(groups, list(train_g))].reset_index(drop=True),
            "valid": full[np.isin(groups, list(valid_g))].reset_index(drop=True),
        }
        if test_g:
            out["test"] = full[np.isin(groups, list(test_g))].reset_index(drop=True)
        return out

    # random
    rng = np.random.default_rng(plan.seed)
    idx = rng.permutation(len(full))
    n_valid = max(1, int(round(len(full) * plan.valid_frac)))
    n_test = int(round(len(full) * plan.test_frac))
    valid_idx = np.sort(idx[:n_valid])
    test_idx = np.sort(idx[n_valid : n_valid + n_test])
    train_idx = np.sort(idx[n_valid + n_test :])
    out = {
        "train": full.iloc[train_idx].reset_index(drop=True),
        "valid": full.iloc[valid_idx].reset_index(drop=True),
    }
    if len(test_idx):
        out["test"] = full.iloc[test_idx].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def frame(task: TaskConfig, split: str) -> pd.DataFrame:
    """The full table for one split. Materialised on demand — used by the profiler."""
    name = normalise_split(split)
    frames = _split_frames(task)
    if name not in frames:
        raise KeyError(f"task {task.name!r} has no split {name!r}; has {sorted(frames)}")
    return frames[name]


def all_frames(task: TaskConfig) -> dict[str, pd.DataFrame]:
    return _split_frames(task)


def _cache_path(task: TaskConfig) -> Path:
    return CACHE_DIR / f"eval-{task.name}-{fingerprint(task)}.npz"


def _build_cache(task: TaskConfig) -> dict[str, EvalArrays]:
    frames = _split_frames(task)
    d = task.data
    payload: dict[str, np.ndarray] = {}
    out: dict[str, EvalArrays] = {}

    for name, df in frames.items():
        if d.target not in df.columns:
            # The held-out split legitimately may not carry labels.
            target = np.full(len(df), np.nan, dtype=np.float64)
        else:
            target = df[d.target].to_numpy()
        groups = df[d.group].to_numpy() if d.group and d.group in df.columns else None
        ids = {c: df[c].to_numpy() for c in d.id_columns if c in df.columns}

        payload[f"{name}__target"] = target
        if groups is not None:
            payload[f"{name}__groups"] = groups
        for c, arr in ids.items():
            payload[f"{name}__id__{c}"] = arr
        out[name] = EvalArrays(name, target, groups, ids)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(task)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    tmp.replace(path)
    return out


def _read_cache(path: Path) -> dict[str, EvalArrays]:
    out: dict[str, EvalArrays] = {}
    with np.load(path, allow_pickle=True) as z:
        names = {k.split("__")[0] for k in z.files}
        for name in sorted(names):
            target = z[f"{name}__target"]
            groups = z[f"{name}__groups"] if f"{name}__groups" in z.files else None
            ids = {
                k.split("__id__")[1]: z[k]
                for k in z.files
                if k.startswith(f"{name}__id__")
            }
            out[name] = EvalArrays(name, target, groups, ids)
    return out


def eval_arrays(task: TaskConfig, split: str) -> EvalArrays:
    """Target, groups and id columns for one split, in row_id order. Cached."""
    name = normalise_split(split)
    path = _cache_path(task)
    if path.is_file():
        try:
            cached = _read_cache(path)
        except (OSError, ValueError, KeyError):
            cached = _build_cache(task)
    else:
        cached = _build_cache(task)
    if name not in cached:
        raise KeyError(f"task {task.name!r} has no split {name!r}; has {sorted(cached)}")
    return cached[name]


def split_names(task: TaskConfig) -> list[str]:
    return sorted(_split_frames(task))


def materialise(task: TaskConfig, *, refresh: bool = False) -> Path:
    """Write each split to its own CSV and return the directory holding them.

    This is what the generated pipeline is pointed at, and it removes a whole class of
    silent failure. If the pipeline had to re-derive a random or grouped split from the
    raw table, it would have to reproduce our permutation exactly — same library, same
    seed handling, same row order — and any drift would score the model against rows it
    had trained on, with nothing anywhere reporting a problem.

    `test.csv` is written **without the target column**. The no-peeking rule then holds
    because the labels are not on disk for the agent to find, rather than because we asked
    it not to look.
    """
    out_dir = CACHE_DIR / f"{task.name}-{fingerprint(task)}"
    marker = out_dir / ".complete"
    if marker.is_file() and not refresh:
        return out_dir

    frames = _split_frames(task)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        if name == "test" and task.data.target in df.columns:
            df = df.drop(columns=[task.data.target])
        df.to_csv(out_dir / f"{name}.csv", index=False)
    marker.write_text("ok\n", encoding="utf-8")
    return out_dir
