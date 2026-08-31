"""Hide the labels a pipeline is not allowed to see, by removing them.

This exists because of a real failure, not a hypothetical one. An official run scored
primary 0.8484 with GAUC 0.99999 — exactly the oracle ceiling, which no model can reach.
The pipeline had computed per-user and per-item `long_view` rates *from the evaluation
split itself* and used them to predict `long_view` on those same rows:

    user_ctr  = eval_rows.groupby('user_id')['long_view'].mean()
    video_ctr = eval_rows.groupby('video_id')['long_view'].mean()

An earlier run's headline result had the same flaw in a milder form. Both were invalid.

The rules make this the disqualifying case rather than a quality problem: on `--split
test` the same code reads the hidden test labels. And it is not something a prompt can be
trusted to prevent — the data card already warned about leakage, and it happened anyway,
twice, because the labels were simply sitting there in the directory we handed over.

`datasource.materialise()` already solves this for generic tasks by writing `test.csv`
without its target column: enforcement by absence. KuaiRand reads the organisers' raw CSVs
directly, so that protection never applied to the benchmark that actually counts. This
module extends the same idea to it.

What is visible depends on the split the pipeline was asked for, and mirrors the rules
exactly:

    --split val   train labels only        (valid and test blanked)
    --split test  train + valid labels     (test blanked)

What is blanked is the label *and every other outcome of the impression* — the click,
the likes, the watch time. Blanking the label alone is not enough and we measured that:
the leaking pipeline still scored 0.84839, because `is_click` and `play_time_ms` carry
the answer just as well. Blanked cells read back as NaN, so a leaking aggregate becomes
NaN rather than a number. Masked copies are built once per
(split, source) and cached, because rewriting ~106 MB of CSV on every iteration would
show up in the wall-clock we report.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

# The organisers' date-range splits. Same constants as vendor/starter_kit/data.py; a
# pipeline that re-derives them gets the same rows, minus the labels it may not have.
RANGES = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
LABEL = "long_view"

# Everything the impression *produced*. These are recorded at the same instant as the
# label, so on an evaluation row they are the future, and blanking the label alone
# achieves nothing: is_click correlates 0.75 with long_view on its own, and
# play_time_ms/duration_ms separates the classes 0.884 against 0.099. A GBDT handed them
# reaches GAUC 0.99999 — the oracle — which is exactly what our first official run did,
# and it kept doing it after we blanked only the label.
#
# The organisers' own baseline uses user_id, video_id, author_id, tab and a duration
# bucket. That is the shape of what is legitimately knowable before the impression, and
# it is why the baseline scores 0.60 rather than 1.00.
POST_OUTCOME = (
    "long_view", "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "is_profile_enter", "play_time_ms", "profile_stay_time",
    "comment_stay_time",
)

# Knowable before the outcome, so left alone: user_id, video_id, date, hourmin, time_ms,
# duration_ms (a property of the video), is_rand (the exposure policy) and tab.

# What a pipeline may see, per split it was asked to produce.
HIDDEN: dict[str, tuple[str, ...]] = {
    "val": ("valid", "test"),
    "valid": ("valid", "test"),
    "test": ("test",),
}


def needs_masking(task) -> bool:
    """Only the starter-kit path. Generic tasks are already protected by materialise()."""
    cfg = getattr(task, "config", None)
    return cfg is not None and getattr(cfg.data, "loader", "") == "starter_kit"


def _fingerprint(src: Path, split: str) -> str:
    parts = [split]
    for name in LOG_FILES:
        p = src / name
        if p.is_file():
            st = p.stat()
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def masked_data_dir(src: Path, split: str, cache_root: Path) -> Path:
    """A copy of `src` in which the labels for `split` are blank. Cached.

    Everything the pipeline might legitimately read — the feature tables, the random
    exposure log — is copied through untouched. Only the label column of the hidden date
    ranges is emptied.
    """
    src = Path(src)
    hide = HIDDEN.get(split)
    if not hide:
        return src

    out = Path(cache_root) / f"masked-{split}-{_fingerprint(src, split)}"
    done = out / ".complete"
    if done.is_file():
        return out
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    lo_hi = [RANGES[s] for s in hide]
    for path in sorted(src.iterdir()):
        if not path.is_file():
            continue
        if path.name not in LOG_FILES:
            shutil.copy2(path, out / path.name)  # features, random log: not labels
            continue
        df = pd.read_csv(path)
        if "date" in df.columns:
            mask = False
            for lo, hi in lo_hi:
                mask = mask | ((df["date"] >= lo) & (df["date"] <= hi))
            for col in POST_OUTCOME:
                if col in df.columns:
                    # Cast first: writing "" into an int64 column is deprecated and
                    # will raise in a later pandas. Object holds the blank, and the
                    # CSV round-trips it back as NaN either way.
                    df[col] = df[col].astype("object")
                    df.loc[mask, col] = ""
        df.to_csv(out / path.name, index=False)

    done.write_text(f"split={split}\nhidden={','.join(hide)}\n", encoding="utf-8")
    return out
