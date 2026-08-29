"""Cached access to the official KuaiRand-Pure splits.

`vendor/starter_kit/data.py::load` re-parses ~2.4M CSV rows on every call. The evaluator
runs at least once per iteration, so we parse once and cache the per-split arrays we need
(`row_id` order, `user_id`, `video_id`, `long_view`) as a single `.npz`.

The cache is keyed by the mtime+size of the two standard log files, so it self-invalidates
if the data underneath ever changes. Row order is exactly `data.load()[split]` order, which
is what `row_id` in a submission indexes into — see `vendor/starter_kit/submit.py`.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
STARTER_KIT = REPO_ROOT / "vendor" / "starter_kit"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "KuaiRand-Pure" / "data"
CACHE_DIR = REPO_ROOT / "data" / "cache"

# The starter kit calls the middle split "valid"; the pipeline CLI contract says "val".
# Accept both everywhere, normalise to the starter kit's spelling.
_SPLIT_ALIASES = {"val": "valid", "valid": "valid", "train": "train", "test": "test"}

LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)


def normalise_split(split: str) -> str:
    """Map 'val' -> 'valid'. Raises on anything unknown."""
    key = split.strip().lower()
    if key not in _SPLIT_ALIASES:
        raise ValueError(f"unknown split {split!r}; expected one of train/val/valid/test")
    return _SPLIT_ALIASES[key]


@dataclass(frozen=True)
class Split:
    """One evaluation split, in `row_id` order."""

    name: str
    user_ids: np.ndarray  # int64, (N,)
    video_ids: np.ndarray  # int64, (N,)
    labels: np.ndarray  # int8, (N,) -- the `long_view` column

    def __len__(self) -> int:
        return int(self.user_ids.shape[0])


def _data_fingerprint(data_dir: Path) -> str:
    """Cheap identity for the underlying CSVs: name, size and mtime of each log file."""
    h = hashlib.sha256()
    for fname in LOG_FILES:
        p = data_dir / fname
        st = p.stat()
        h.update(f"{fname}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _load_via_starter_kit(data_dir: Path) -> dict[str, list]:
    """Call the vendored `data.load()` unmodified, so row order is authoritative."""
    sys.path.insert(0, str(STARTER_KIT))
    try:
        import data as starter_data  # type: ignore[import-not-found]

        return starter_data.load(str(data_dir))
    finally:
        # Leave sys.path as we found it; the starter kit's module name is very generic.
        if sys.path and sys.path[0] == str(STARTER_KIT):
            sys.path.pop(0)
        sys.modules.pop("data", None)


def _build_cache(data_dir: Path, cache_path: Path) -> dict[str, Split]:
    raw = _load_via_starter_kit(data_dir)
    payload: dict[str, np.ndarray] = {}
    splits: dict[str, Split] = {}
    for name, rows in raw.items():
        # data.load() rows are (date, user_id, video_id, author_id, tab, duration_ms, label)
        users = np.fromiter((int(r[1]) for r in rows), dtype=np.int64, count=len(rows))
        videos = np.fromiter((int(r[2]) for r in rows), dtype=np.int64, count=len(rows))
        labels = np.fromiter((int(r[6]) for r in rows), dtype=np.int8, count=len(rows))
        payload[f"{name}__user_ids"] = users
        payload[f"{name}__video_ids"] = videos
        payload[f"{name}__labels"] = labels
        splits[name] = Split(name, users, videos, labels)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Write through a handle: np.savez would otherwise append a second ".npz" to the name.
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp, cache_path)
    return splits


def _read_cache(cache_path: Path) -> dict[str, Split]:
    with np.load(cache_path) as z:
        names = {k.split("__", 1)[0] for k in z.files}
        return {
            n: Split(n, z[f"{n}__user_ids"], z[f"{n}__video_ids"], z[f"{n}__labels"])
            for n in names
        }


_MEM_CACHE: dict[tuple[str, str], dict[str, Split]] = {}


def load_splits(data_dir: Path | str | None = None) -> dict[str, Split]:
    """Return {'train','valid','test'} -> Split, memoised in-process and on disk."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"KuaiRand-Pure data dir not found: {data_dir}\n"
            "Download it with:\n"
            "  curl -L -o data/KuaiRand-Pure.tar.gz "
            "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz\n"
            "  tar xzf data/KuaiRand-Pure.tar.gz -C data/"
        )

    fp = _data_fingerprint(data_dir)
    mem_key = (str(data_dir), fp)
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]

    cache_path = CACHE_DIR / f"splits-{fp}.npz"
    splits = _read_cache(cache_path) if cache_path.exists() else _build_cache(data_dir, cache_path)
    _MEM_CACHE[mem_key] = splits
    return splits


def get_split(split: str, data_dir: Path | str | None = None) -> Split:
    """Return one split by name, accepting either 'val' or 'valid'."""
    return load_splits(data_dir)[normalise_split(split)]
