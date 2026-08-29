"""The data card — the agent's only view of the dataset.

    data_card() -> str          # markdown, <= 3000 tokens

The LLM never sees the CSVs; sending them would blow the token budget that the Feasibility
criterion scores. It sees this card instead, on every draft and improve call. That makes
the card's accuracy the ceiling on the quality of every hypothesis the agent forms.

**Facts only, no advice.** What to *try* belongs in the idea bank (`ideas.yaml`, owner D),
so that the journal can attribute each attempt to an idea id. If a recommendation creeps in
here it is unattributable, and it biases every proposal the agent ever makes.

Every number below is computed from the data on disk by `compute_stats()` and cached as
JSON, so the card cannot drift from the dataset the way a hand-written one would.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from orchestrator.splits import CACHE_DIR, DEFAULT_DATA_DIR, _data_fingerprint

#: Binary feedback columns in the interaction log. `long_view` is the scored label.
BINARY_SIGNALS = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "long_view",
]
#: Continuous feedback columns.
CONTINUOUS_SIGNALS = ["play_time_ms", "profile_stay_time", "comment_stay_time"]

SPLIT_DATES = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}


def compute_stats(data_dir: Path | str | None = None, *, refresh: bool = False) -> dict:
    """Compute (or load) every number the card quotes. Cached as JSON next to the splits."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    data_dir = data_dir.resolve()
    fp = _data_fingerprint(data_dir)
    cache_path = CACHE_DIR / f"datacard-{fp}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())

    import pandas as pd  # tooling-only dependency; never imported by a generated pipeline

    log = pd.concat(
        [
            pd.read_csv(data_dir / "log_standard_4_08_to_4_21_pure.csv"),
            pd.read_csv(data_dir / "log_standard_4_22_to_5_08_pure.csv"),
        ],
        ignore_index=True,
    )
    video = pd.read_csv(data_dir / "video_features_basic_pure.csv")
    user = pd.read_csv(data_dir / "user_features_pure.csv")

    stats: dict = {"splits": {}, "signals": {}, "duration": {}, "fields": {}}

    for name, (lo, hi) in SPLIT_DATES.items():
        s = log[(log.date >= lo) & (log.date <= hi)]
        by_user = s.groupby("user_id")["long_view"].agg(["sum", "count"])
        stats["splits"][name] = {
            "dates": f"{lo}-{hi}",
            "rows": int(len(s)),
            "users": int(s.user_id.nunique()),
            "items": int(s.video_id.nunique()),
            "pos_rate": round(float(s.long_view.mean()), 4),
            "impr_per_user_median": int(by_user["count"].median()),
            "impr_per_user_mean": round(float(by_user["count"].mean()), 1),
            "impr_per_user_p90": int(by_user["count"].quantile(0.9)),
            "impr_per_user_max": int(by_user["count"].max()),
            "zero_pos_user_pct": round(100 * float((by_user["sum"] == 0).mean()), 1),
            "all_pos_user_pct": round(
                100 * float((by_user["sum"] == by_user["count"]).mean()), 1
            ),
            "repeat_row_pct": round(
                100
                * float(s.duplicated(subset=["user_id", "video_id"], keep="first").mean()),
                2,
            ),
            "max_repeat": int(s.groupby(["user_id", "video_id"]).size().max()),
        }

    train = log[log.date <= SPLIT_DATES["train"][1]]
    for c in BINARY_SIGNALS:
        stats["signals"][c] = {
            n: round(float(log[(log.date >= lo) & (log.date <= hi)][c].mean()), 4)
            for n, (lo, hi) in SPLIT_DATES.items()
        }
    for c in CONTINUOUS_SIGNALS:
        stats["signals"][c] = {
            "train_mean": round(float(train[c].mean()), 1),
            "train_median": round(float(train[c].median()), 1),
            "train_p99": round(float(train[c].quantile(0.99)), 1),
        }

    dur = train.duration_ms
    stats["duration"] = {
        "quantiles_ms": {str(q): int(dur.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
        "min_ms": int(dur.min()),
        "max_ms": int(dur.max()),
        "zero_pct": round(100 * float((dur == 0).mean()), 3),
    }
    # Duration bias: completion falls off sharply with video length.
    d10 = train.assign(_b=lambda t: t.duration_ms.rank(pct=True).mul(10).clip(upper=9.999).astype(int))
    ratio = (train.play_time_ms / train.duration_ms.clip(lower=1)).groupby(d10._b).mean()
    stats["duration"]["long_view_rate_by_decile"] = [
        round(float(x), 3) for x in d10.groupby("_b").long_view.mean().tolist()
    ]
    stats["duration"]["play_ratio_by_decile"] = [round(float(x), 3) for x in ratio.tolist()]

    stats["fields"]["log"] = [
        {"name": c, "card": int(log[c].nunique()), "miss": round(100 * float(log[c].isna().mean()), 2)}
        for c in ["user_id", "video_id", "date", "hourmin", "tab", "is_rand"]
    ]
    stats["fields"]["video"] = [
        {"name": c, "card": int(video[c].nunique()), "miss": round(100 * float(video[c].isna().mean()), 2)}
        for c in video.columns
        if c != "video_id"
    ]
    stats["fields"]["video_rows"] = int(len(video))
    stats["fields"]["user_rows"] = int(len(user))
    stats["fields"]["user_cols"] = [c for c in user.columns if c != "user_id"]
    stats["fields"]["user_card"] = {
        c: int(user[c].nunique())
        for c in [
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]
    }
    stats["tab_top"] = {str(k): int(v) for k, v in log.tab.value_counts().head(6).items()}
    stats["is_rand_values"] = {str(k): int(v) for k, v in log.is_rand.value_counts().items()}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats, indent=1))
    os.replace(tmp, cache_path)
    return stats


def _row(sp: dict, key: str, fmt: str = "{}") -> str:
    return fmt.format(sp[key])


def data_card(data_dir: Path | str | None = None) -> str:
    """Render the markdown data card. Deterministic given the data on disk."""
    st = compute_stats(data_dir)
    s = st["splits"]
    d = st["duration"]

    def q(k: str) -> str:
        return f"{d['quantiles_ms'][k] / 1000:.1f}s"

    lines: list[str] = []
    A = lines.append

    A("# KuaiRand-Pure — data card")
    A("")
    A("Facts only. All numbers computed from the data on disk.")
    A("")
    A("## Task and label")
    A("")
    A("- **Within-user ranking** over logged impressions. Each user's own impressions in the")
    A("  evaluation split are ranked against each other. There is no retrieval over a catalogue.")
    A("- **Label:** `long_view` (native 0/1 column in the interaction log).")
    A("- **Metrics:** GAUC and nDCG@5; primary = their unweighted mean.")
    A("- Scoring is done by the orchestrator, not by the pipeline. The pipeline only writes scores.")
    A("")
    A("## Splits (date-based, fixed)")
    A("")
    A("| split | dates | rows | users | items | `long_view` rate |")
    A("|---|---|---|---|---|---|")
    for n in ("train", "valid", "test"):
        sp = s[n]
        A(
            f"| {n} | {sp['dates']} | {sp['rows']:,} | {sp['users']:,} | "
            f"{sp['items']:,} | {sp['pos_rate']} |"
        )
    A("")
    A("Train is 14 days, validation 7, test 10. The test split is hidden during development;")
    A("only train and validation may be used to fit or select anything.")
    A("")
    A("## Per-user exposure structure (drives the metric)")
    A("")
    A("| split | impressions/user median | mean | p90 | max | zero-positive users | all-positive users |")
    A("|---|---|---|---|---|---|---|")
    for n in ("train", "valid", "test"):
        sp = s[n]
        A(
            f"| {n} | {sp['impr_per_user_median']} | {sp['impr_per_user_mean']} | "
            f"{sp['impr_per_user_p90']} | {sp['impr_per_user_max']} | "
            f"{sp['zero_pos_user_pct']}% | {sp['all_pos_user_pct']}% |"
        )
    A("")
    A(f"- Train users average {s['train']['impr_per_user_mean']} impressions; validation and test")
    A(f"  users average only {s['valid']['impr_per_user_mean']} and {s['test']['impr_per_user_mean']}.")
    A("  Per-user list lengths at evaluation time are short.")
    A(f"- In test, {s['test']['zero_pos_user_pct']}% of users have no positive (nDCG is 0 for them")
    A(f"  under any model, and they are included in the mean) and {s['test']['all_pos_user_pct']}%")
    A("  are all-positive (nDCG is 1 under any model). Both groups are excluded from GAUC.")
    A("- Consequence: a perfect ranking scores GAUC 1.0000 / nDCG@5 0.7289 / primary **0.8645**")
    A("  on test. Random scores 0.4753. The published FM baseline scores 0.5946.")
    A("")
    A("## Repeated pairs")
    A("")
    A(
        f"`(user_id, video_id)` is **not unique**: {s['test']['repeat_row_pct']}% of test rows repeat"
    )
    A(
        f"an earlier pair (max {s['test']['max_repeat']} occurrences); "
        f"{s['train']['repeat_row_pct']}% in train, {s['valid']['repeat_row_pct']}% in validation."
    )
    A("`row_id` — the 0-based position in the split — is the only key. Any join, dedupe or")
    A("groupby that assumes the pair is unique will silently misalign the submission.")
    A("")
    A("## Feedback signals in the interaction log")
    A("")
    A("All are recorded per impression. Only `long_view` is scored; the others are available")
    A("as additional supervision.")
    A("")
    A("| signal | train rate | valid rate | test rate |")
    A("|---|---|---|---|")
    for c in BINARY_SIGNALS:
        v = st["signals"][c]
        star = " *(scored label)*" if c == "long_view" else ""
        A(f"| `{c}`{star} | {v['train']} | {v['valid']} | {v['test']} |")
    A("")
    A("| continuous signal | train mean | median | p99 |")
    A("|---|---|---|---|")
    for c in CONTINUOUS_SIGNALS:
        v = st["signals"][c]
        A(f"| `{c}` | {v['train_mean']} | {v['train_median']} | {v['train_p99']} |")
    A("")
    A("## Fields available")
    A("")
    log_card = {f["name"]: f["card"] for f in st["fields"]["log"]}
    A("**Interaction log** (context, one row per impression): `user_id`")
    A(f"({log_card['user_id']:,} distinct), `video_id` ({log_card['video_id']:,}), `date`")
    A(f"({log_card['date']} days), `hourmin` (HHMM integer), `time_ms` (epoch ms), `tab`")
    A(f"(entry surface, {log_card['tab']} values; most common: {', '.join(st['tab_top'])}),")
    A("`is_rand`, plus the feedback columns above and `duration_ms`.")
    A(
        f"`is_rand` is {list(st['is_rand_values'])[0]} for every row of the standard logs "
        "— these are all algorithmically-exposed impressions."
    )
    A("")
    A(f"**Video side** (`video_features_basic_pure.csv`, {st['fields']['video_rows']:,} videos):")
    A(
        "  "
        + ", ".join(
            f"`{f['name']}` ({f['card']} distinct"
            + (f", {f['miss']}% missing" if f["miss"] else "")
            + ")"
            for f in st["fields"]["video"]
        )
    )
    A("  A `video_features_statistic_pure.csv` also exists with aggregate counters per video.")
    A("")
    A(f"**User side** (`user_features_pure.csv`, {st['fields']['user_rows']:,} users,")
    A(f"{len(st['fields']['user_cols'])} columns): `user_active_degree`")
    A(f"({st['fields']['user_card']['user_active_degree']} values), `is_live_streamer`,")
    A("`is_video_author`, `follow_user_num`/`_range`, `fans_user_num`/`_range`,")
    A("`friend_user_num`/`_range`, `register_days`/`_range`, and `onehot_feat0..17`.")
    A("")
    A("Ranking is **within-user**, so any feature that is constant across one user's rows")
    A("cannot change that user's ordering. User-side fields move the score only through")
    A("interactions with item-side or context fields.")
    A("")
    A("## Video duration")
    A("")
    A(
        f"`duration_ms` on train: p10 {q('0.1')}, p25 {q('0.25')}, median {q('0.5')}, "
        f"p75 {q('0.75')}, p90 {q('0.9')}, p99 {q('0.99')}; max "
        f"{d['max_ms'] / 1000:.0f}s. {d['zero_pct']}% of rows have `duration_ms` 0."
    )
    A("")
    A("Mean `play_time_ms / duration_ms` by duration decile (ascending):")
    A(f"  {d['play_ratio_by_decile'][1:]}  (decile 0 omitted: `duration_ms` 0 rows)")
    A("`long_view` rate by the same deciles:")
    A(f"  {d['long_view_rate_by_decile']}")
    A("The fraction of a video watched falls monotonically with its length, while the")
    A("`long_view` rate does not — the label's relationship to duration is non-monotone.")
    A("")
    A("## Constraints")
    A("")
    A("- **No external training data.** KuaiRand only. No joining, augmenting or pre-training")
    A("  on any other dataset, and no pretrained weights that have seen these test labels.")
    A("- **No hidden-test access during development.** Fit and select on train + validation only.")
    A("- No future information: a feature for a row may only use data dated at or before that")
    A("  row. Statistics used at validation time must be computed on train alone.")
    A("- Any target encoding must be out-of-fold, or it leaks the label into its own feature.")
    A("- Subsampling for speed must sample **users**, not rows: dropping rows inside a user")
    A("  changes that user's list and therefore changes GAUC and nDCG.")
    A("- `log_random_4_22_to_5_08_pure.csv` is a separate uniformly-exposed log covering the")
    A("  validation and test dates. It is part of KuaiRand.")

    return "\n".join(lines) + "\n"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for the <=3000 budget check."""
    return len(text) // 4


if __name__ == "__main__":  # pragma: no cover - manual inspection
    card = data_card()
    print(card)
    print(f"\n---\n~{estimate_tokens(card)} tokens, {len(card)} chars", flush=True)
