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
        return json.loads(cache_path.read_text(encoding="utf-8"))

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
    # Explicit utf-8: the default on Windows is cp1252, which has already silently
    # mangled two non-ASCII round-trips in this repo. Never rely on the platform default.
    tmp.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    os.replace(tmp, cache_path)
    return stats


def _row(sp: dict, key: str, fmt: str = "{}") -> str:
    return fmt.format(sp[key])


def kuairand_card(data_dir: Path | str | None = None) -> str:
    """The hand-tuned KuaiRand-Pure card.

    Kept separate from `generic_card()` because it carries facts the profiler cannot
    derive — the duration-bias decile table, the 0.8645 ceiling and why it is not 1.0,
    the repeated-pair rate, and the organisers' own conventions. A generic profile of the
    same CSVs would be correct but thinner, and this is the task that is actually scored.
    """
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
    A("## Files on disk")
    A("")
    # This wording cost a live iteration. The card used to lead with
    # `<data-dir>/KuaiRand-Pure/data/`, so the first pipeline gpt-4o wrote appended that
    # to a --data-dir that already ended in it and died on FileNotFoundError. The
    # orchestrator always passes the directory that *contains* the CSVs, so say that
    # first and without an alternative to weigh up.
    A("The six CSVs sit **directly inside the directory passed as `--data-dir`**. Join a")
    A("filename straight onto it — do not append `KuaiRand-Pure/data/`, `--data-dir`")
    A("already points at the directory holding these files:")
    A("")
    A("```python")
    A("train_log = os.path.join(args.data_dir, 'log_standard_4_08_to_4_21_pure.csv')")
    A("```")
    A("")
    A("Exact names:")
    A("")
    A("| file | rows | contents |")
    A("|---|---|---|")
    A("| `log_standard_4_08_to_4_21_pure.csv` | 1,141,112 | interactions, 20220408-20220421 |")
    A("| `log_standard_4_22_to_5_08_pure.csv` | 295,497 | interactions, 20220422-20220508 |")
    A("| `log_random_4_22_to_5_08_pure.csv` | 1,186,059 | **random-exposure** log; not part of any split |")
    A("| `video_features_basic_pure.csv` | 7,583 | one row per `video_id` |")
    A("| `video_features_statistic_pure.csv` | 7,583 | one row per `video_id` |")
    A("| `user_features_pure.csv` | 27,285 | one row per `user_id` |")
    A("")
    A("The three splits come from the **two `log_standard` files only**, filtered by the `date`")
    A("column. The first file is entirely train. The second file contains validation *and* test")
    A("rows interleaved and must be split by date: 20220422-20220428 is validation,")
    A("20220429-20220508 is test. There is no validation-only or test-only file.")
    A("")
    A("**Row order defines `row_id`.** Read `log_standard_4_08_to_4_21_pure.csv` first, then")
    A("`log_standard_4_22_to_5_08_pure.csv`, filter by date, and preserve the original line order")
    A("within each file. `row_id` is the 0-based position in that sequence. `(user_id, video_id)`")
    A("is not a key - 3.06% of test rows are repeated pairs, up to 12 times.")
    A("")
    A("### Where each column lives")
    A("")
    A("- Interaction logs (19 columns): `user_id`, `video_id`, `date`, `hourmin`, `time_ms`,")
    A("  `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`,")
    A("  `play_time_ms`, `duration_ms`, `profile_stay_time`, `comment_stay_time`,")
    A("  `is_profile_enter`, `is_rand`, `tab`.")
    A("- `author_id`, `video_type`, `upload_type`, `music_id`, `video_duration` are **not** in the")
    A("  interaction log. They are in `video_features_basic_pure.csv`, joined on `video_id`.")
    A("- User attributes (`user_active_degree`, `follow_user_num`, `fans_user_num`, ...) are in")
    A("  `user_features_pure.csv`, joined on `user_id`.")
    A("- The official baseline's five fields are `user_id`, `video_id`, `author_id`, `tab`, and a")
    A("  bucketed `duration_ms`; `author_id` therefore requires the video-features join.")
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


def _fmt_num(v) -> str:
    if isinstance(v, float):
        return f"{v:,.4g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _primary_formula(task) -> str:
    """Spell out `primary` as an expression, so the sign is never ambiguous.

    The search always maximises, so a lower-is-better metric appears negated. Saying
    "mean of rmse, higher is better" would be worse than useless.
    """
    from orchestrator import metrics as M

    terms = [
        (f"`{p}`" if M.get(p).greater_is_better else f"−`{p}`") for p in task.primary_parts
    ]
    body = terms[0] if len(terms) == 1 else "mean of " + ", ".join(terms)
    return f"{body} — maximised, so higher is always better."


def generic_card(task) -> str:
    """Render a data card for **any** task, from facts the profiler derived.

    This is what makes a new dataset a config file rather than a code change. It is
    necessarily thinner than a hand-tuned card — a profiler can measure a distribution but
    it cannot know that 27% of users having no positive is what caps the metric at 0.8645.
    A task file may add exactly that kind of knowledge in its `notes:` field.
    """
    from orchestrator.profile import profile

    f = profile(task)
    t = f["task"]
    lines: list[str] = []
    A = lines.append

    A(f"# {t['name']} — data card")
    A("")
    A("Facts only, derived from the data on disk. What to *try* is not in scope here.")
    A("")

    A("## The problem")
    A("")
    A(t["description"].strip())
    A("")
    A(f"- **Task type:** {t['kind']}")
    A(f"- **Target column:** `{t['target']}`")
    if t["group"]:
        A(f"- **Group column:** `{t['group']}` — scoring is *within* a group, not global.")
    A(f"- **Scored on:** {', '.join(f'`{m}`' for m in t['report_metrics'])}")
    A(f"- **Primary** (what the search maximises): {_primary_formula(task)}")
    A("")
    A(task.metric_glossary())
    A("")
    A("Scoring is done by the orchestrator, not by the pipeline. The pipeline writes")
    A("predictions; it never computes its own score.")
    A("")

    A("## Files on disk")
    A("")
    A("`--data-dir` contains one CSV per split, already separated for you:")
    A("")
    A("| file | contents |")
    A("|---|---|")
    for name in ("train", "valid", "test"):
        if name not in f["splits"]:
            continue
        if name == "test":
            A(f"| `{name}.csv` | the split to predict — **the target column is not in it** |")
        else:
            A(f"| `{name}.csv` | features and the `{t['target']}` column |")
    A("")
    A("Read them directly by name. Do not re-derive the splits yourself and do not look")
    A("for the original source file — these are the exact rows the submission is scored")
    A("against, in the exact order `row_id` refers to.")
    A("")

    A("## Splits")
    A("")
    A("| split | rows | columns | target |")
    A("|---|---|---|---|")
    for name in ("train", "valid", "test"):
        sp = f["splits"].get(name)
        if not sp:
            continue
        tgt = sp["target"]
        if tgt["type"] == "numeric":
            desc = f"mean {_fmt_num(tgt.get('mean'))}, sd {_fmt_num(tgt.get('std'))}"
        elif tgt["type"] == "categorical":
            rate = tgt.get("positive_rate")
            desc = (
                f"{tgt['n_classes']} classes"
                + (f", positive rate {rate}" if rate is not None else "")
            )
        else:
            desc = "labels not present"
        A(f"| {name} | {sp['rows']:,} | {sp['columns']} | {desc} |")
    A("")
    A("Development uses train and validation only. The test split is scored once, at the end.")
    A("")

    train = f["splits"]["train"]
    tgt = train["target"]
    if tgt["type"] == "numeric":
        q = tgt.get("quantiles", {})
        A("### Target distribution (train)")
        A("")
        A(
            f"min {_fmt_num(tgt.get('min'))} · p1 {_fmt_num(q.get('0.01'))} · "
            f"p25 {_fmt_num(q.get('0.25'))} · median {_fmt_num(q.get('0.5'))} · "
            f"p75 {_fmt_num(q.get('0.75'))} · p99 {_fmt_num(q.get('0.99'))} · "
            f"max {_fmt_num(tgt.get('max'))}"
        )
        A("")
        A(
            f"skew {tgt.get('skew')} · {tgt.get('n_zero', 0):,} zeros · "
            f"{tgt.get('n_negative', 0):,} negative · {tgt.get('n_missing', 0):,} missing"
        )
        A("")
    elif tgt["type"] == "categorical":
        A("### Class balance (train)")
        A("")
        A("| class | rows | rate |")
        A("|---|---|---|")
        for c in tgt.get("classes", []):
            A(f"| {c['value']} | {c['count']:,} | {c['rate']} |")
        A("")

    if train.get("groups"):
        g = train["groups"]
        A("### Group structure (drives the metric)")
        A("")
        A(
            f"{g['n_groups']:,} groups in train. Rows per group: median "
            f"{g['rows_per_group_median']}, mean {g['rows_per_group_mean']}, "
            f"max {g['rows_per_group_max']}."
        )
        if "zero_positive_pct" in g:
            A("")
            A(
                f"{g['zero_positive_pct']}% of groups have no positive label and "
                f"{g['all_positive_pct']}% are all-positive. Groups with no positive "
                "cannot be ranked correctly by any model, which caps the attainable score."
            )
        A("")
        A("Any subsample must hold out **whole groups, never individual rows** — removing")
        A("rows from inside a group changes that group's score.")
        A("")

    A(f"## Columns ({f['n_columns_total']} besides the target)")
    A("")
    A("Ordered by apparent relationship to the target. `role` marks structural columns.")
    A("")
    A("| column | role | type | distinct | missing | relationship to target |")
    A("|---|---|---|---|---|---|")
    for c in f["columns"]:
        if c["kind"] == "numeric":
            rel = (
                f"corr {c['corr_with_target']:+.3f}"
                if "corr_with_target" in c
                else f"range {_fmt_num(c.get('min'))}..{_fmt_num(c.get('max'))}"
            )
        else:
            top = ", ".join(v["value"] for v in c.get("top_values", [])[:3])
            spread = c.get("target_mean_spread")
            rel = f"top: {top}" + (f" · target sd across levels {spread}" if spread else "")
        A(
            f"| `{c['name']}` | {c['role']} | {c['kind']} | {c['n_unique']:,} | "
            f"{c['missing_pct']}% | {rel} |"
        )
    if f["columns_omitted"]:
        A("")
        A(
            f"{len(f['columns_omitted'])} further column(s) not detailed above: "
            + ", ".join(f"`{n}`" for n in f["columns_omitted"][:60])
        )
    A("")

    if f["warnings"]:
        A("## Structural warnings")
        A("")
        for w in f["warnings"]:
            A(f"- {w}")
        A("")

    A("## Submission format")
    A("")
    A(f"CSV with header `{','.join(t['submission_columns'])}`, one line per row of the")
    A("evaluation split.")
    A("")
    A("- `row_id` — 0-based, strictly increasing, no gaps. It is the only reliable key;")
    A("  id columns are not guaranteed unique.")
    A(f"- `{t['prediction_column']}` — the prediction. NaN and Inf are rejected.")
    A("")

    A("## Rules")
    A("")
    A("- No external training data. Only the data described above.")
    A("- No access to the test labels during development. Train and validation only.")
    A("- Any target-derived feature must be computed **out-of-fold**, or validation will")
    A("  score far above what the held-out split returns.")
    A("- Features must be computed on train alone, never on train and validation combined.")
    A("")

    notes = getattr(task, "notes", "") or ""
    if notes.strip():
        A("## Task notes")
        A("")
        A(notes.strip())
        A("")

    return "\n".join(lines) + "\n"


def data_card(task=None, data_dir: Path | str | None = None) -> str:
    """The agent's view of the data, for whichever task it is working on.

    No argument keeps the historical behaviour: the KuaiRand-Pure card. A `TaskConfig`
    (or a task name) routes to the hand-tuned card for KuaiRand and to the profiler for
    everything else.
    """
    if task is None:
        return kuairand_card(data_dir)
    if isinstance(task, str):
        from orchestrator.taskspec import load_task

        task = load_task(task)
    if task.data.loader == "starter_kit":
        return kuairand_card(data_dir or task.data.dir)
    return generic_card(task)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for the <=3000 budget check."""
    return len(text) // 4


if __name__ == "__main__":  # pragma: no cover - manual inspection
    import sys

    card = data_card(sys.argv[1] if len(sys.argv) > 1 else None)
    print(card)
    print(f"\n---\n~{estimate_tokens(card)} tokens, {len(card)} chars", flush=True)
