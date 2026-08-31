"""Task configuration — what turns this from a KuaiRand agent into an ML agent.

    load_task("tasks/kuairand-pure.yaml") -> TaskConfig
    load_task("house-prices")             -> TaskConfig   (looked up in tasks/)

A task file is the *only* thing a user has to write to point the agent at a new problem.
It names the data, the target, how to split, what to optimise, and what a submission looks
like. Everything downstream — the data card, the evaluator, the prompts, the report — reads
its facts from here instead of from a constant somewhere in the code.

Design rule: **a task file describes, it never instructs.** No hyperparameters, no model
suggestions, no "try gradient boosting". What to try is the agent's job, informed by the
idea bank. If a field here started telling the agent how to solve the problem, we would be
solving it for them and the Innovation criterion would be measuring us, not the agent.

Minimal example — this is enough to run:

    name: house-prices
    kind: regression
    description: Predict the sale price of a house from its attributes.
    data:
      dir: data/house-prices
      file: train.csv
      target: SalePrice
      split: {strategy: random, valid_frac: 0.2}
    metrics:
      primary: [rmse]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from orchestrator import metrics as M

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"

TaskKind = Literal["regression", "binary", "multiclass", "ranking"]
SplitStrategy = Literal["predefined", "random", "date", "group"]

VALID_KINDS = ("regression", "binary", "multiclass", "ranking")
VALID_STRATEGIES = ("predefined", "random", "date", "group")


class TaskConfigError(ValueError):
    """A task file that cannot be used. The message names the field and what to do."""


@dataclass(frozen=True)
class SplitPlan:
    strategy: SplitStrategy = "random"
    valid_frac: float = 0.2
    test_frac: float = 0.0
    seed: int = 0
    date_column: str | None = None
    #: For `date`: {"train": [lo, hi], "valid": [lo, hi], "test": [lo, hi]} inclusive.
    ranges: dict[str, list[Any]] = field(default_factory=dict)
    #: For `group`: hold out whole groups, never rows, so grouped metrics stay meaningful.
    group_column: str | None = None


@dataclass(frozen=True)
class DataSpec:
    dir: Path
    loader: str = "tabular"  # "tabular" | "starter_kit"
    files: dict[str, str] = field(default_factory=dict)
    file: str | None = None
    target: str = ""
    group: str | None = None
    id_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()
    split: SplitPlan = field(default_factory=SplitPlan)


@dataclass(frozen=True)
class TaskConfig:
    name: str
    kind: TaskKind
    description: str
    data: DataSpec
    primary_parts: tuple[str, ...]
    report_metrics: tuple[str, ...]
    submission_columns: tuple[str, ...]
    prediction_column: str = "prediction"
    baseline_val: dict[str, float] = field(default_factory=dict)
    baseline_test: dict[str, float] = field(default_factory=dict)
    ceiling: float | None = None
    max_iters: int = 50
    wall_clock_s: int = 6 * 3600
    conv_eps: float = 0.002
    conv_n: int = 3
    #: Flat scored iterations before the policy explores instead of improving the
    #: best node. Validated to be < `conv_n`; see `TaskSpec.explore_after`.
    explore_after: int = 2
    seed_std: float | None = None
    #: Idea bank for this task. Domain knowledge is per-problem — a bank of recommender
    #: ideas is worse than useless on a regression task, because the agent will try them.
    ideas_path: Path | None = None
    #: Free-form facts a profiler cannot derive, appended to the generated data card.
    notes: str = ""
    source_path: Path | None = None

    # -- convenience -------------------------------------------------------

    @property
    def all_metrics(self) -> tuple[str, ...]:
        """Every metric we compute: the report set plus anything primary needs."""
        seen: list[str] = []
        for m in (*self.report_metrics, *self.primary_parts):
            if m not in seen:
                seen.append(m)
        return tuple(seen)

    @property
    def needs_groups(self) -> bool:
        return any(M.get(m).needs_groups for m in self.all_metrics)

    def primary(self, metrics: dict[str, float]) -> float:
        """Composite, always higher-is-better. Falls back to a stored value if present."""
        if "primary" in metrics:
            return float(metrics["primary"])
        return M.primary_of(metrics, self.primary_parts)

    def metric_glossary(self) -> str:
        """One line per metric, for the data card and the prompt."""
        out = []
        for name in self.all_metrics:
            m = M.get(name)
            arrow = "higher is better" if m.greater_is_better else "LOWER is better"
            desc = f" — {m.description}" if m.description else ""
            out.append(f"- `{name}` ({arrow}){desc}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _require(d: dict, key: str, where: str) -> Any:
    if key not in d or d[key] in (None, ""):
        raise TaskConfigError(f"{where}: missing required field {key!r}")
    return d[key]


def _as_tuple(v: Any) -> tuple[str, ...]:
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


def parse_task(raw: dict, *, source: Path | None = None) -> TaskConfig:
    """Turn a parsed task file into a validated `TaskConfig`."""
    where = str(source) if source else "task config"
    if not isinstance(raw, dict):
        raise TaskConfigError(f"{where}: top level must be a mapping")

    name = str(_require(raw, "name", where))
    kind = str(raw.get("kind", "regression")).lower()
    if kind not in VALID_KINDS:
        raise TaskConfigError(f"{where}: kind={kind!r}; expected one of {VALID_KINDS}")

    description = str(raw.get("description") or "").strip()
    if not description:
        raise TaskConfigError(
            f"{where}: 'description' is required — it is the problem statement the agent "
            "reads before writing any code. Two or three sentences on what is being "
            "predicted and why."
        )

    d = raw.get("data") or {}
    if not isinstance(d, dict):
        raise TaskConfigError(f"{where}: 'data' must be a mapping")

    data_dir = Path(str(_require(d, "dir", f"{where}.data")))
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    sp_raw = d.get("split") or {}
    strategy = str(sp_raw.get("strategy", "predefined" if d.get("files") else "random")).lower()
    if strategy not in VALID_STRATEGIES:
        raise TaskConfigError(
            f"{where}.data.split: strategy={strategy!r}; expected one of {VALID_STRATEGIES}"
        )
    if strategy == "date" and not sp_raw.get("date_column"):
        raise TaskConfigError(f"{where}.data.split: strategy 'date' needs 'date_column'")
    if strategy == "date" and not sp_raw.get("ranges"):
        raise TaskConfigError(
            f"{where}.data.split: strategy 'date' needs 'ranges', e.g. "
            "{train: [20220408, 20220421], valid: [20220422, 20220428]}"
        )

    split = SplitPlan(
        strategy=strategy,  # type: ignore[arg-type]
        valid_frac=float(sp_raw.get("valid_frac", 0.2)),
        test_frac=float(sp_raw.get("test_frac", 0.0)),
        seed=int(sp_raw.get("seed", 0)),
        date_column=sp_raw.get("date_column"),
        ranges={k: list(v) for k, v in (sp_raw.get("ranges") or {}).items()},
        group_column=sp_raw.get("group_column") or d.get("group"),
    )

    loader = str(d.get("loader", "tabular")).lower()
    files = {str(k): str(v) for k, v in (d.get("files") or {}).items()}
    single = d.get("file")
    if loader == "tabular" and not files and not single:
        raise TaskConfigError(
            f"{where}.data: give either 'file' (one table, split by 'split') or "
            "'files' (a mapping like {train: train.csv, test: test.csv})"
        )
    if strategy == "predefined" and not files:
        raise TaskConfigError(
            f"{where}.data: split strategy 'predefined' needs 'files' naming each split"
        )

    data = DataSpec(
        dir=data_dir,
        loader=loader,
        files=files,
        file=str(single) if single else None,
        target=str(_require(d, "target", f"{where}.data")),
        group=d.get("group"),
        id_columns=_as_tuple(d.get("id_columns")),
        drop_columns=_as_tuple(d.get("drop_columns")),
        split=split,
    )

    m = raw.get("metrics") or {}
    primary_parts = _as_tuple(m.get("primary"))
    if not primary_parts:
        raise TaskConfigError(
            f"{where}.metrics: 'primary' is required — one or more metric names whose mean "
            f"the agent optimises. Known: {', '.join(M.available())}, plus ndcg@K / map@K."
        )
    report_metrics = _as_tuple(m.get("report")) or primary_parts
    for metric_name in {*primary_parts, *report_metrics}:
        try:
            met = M.get(metric_name)
        except KeyError as exc:
            raise TaskConfigError(f"{where}.metrics: {exc}") from None
        if met.needs_groups and not data.group:
            raise TaskConfigError(
                f"{where}: metric {metric_name!r} is a grouped metric, so data.group must "
                "name the column that defines a group (e.g. user_id)"
            )

    sub = raw.get("submission") or {}
    pred_col = str(sub.get("prediction_column", "prediction"))
    sub_cols = _as_tuple(sub.get("columns"))
    if not sub_cols:
        sub_cols = ("row_id", *data.id_columns, pred_col)
    if "row_id" not in sub_cols:
        raise TaskConfigError(
            f"{where}.submission.columns must include 'row_id' — it is the only reliable key. "
            "Id columns are not guaranteed unique."
        )
    if pred_col not in sub_cols:
        raise TaskConfigError(
            f"{where}.submission: prediction_column {pred_col!r} is not in columns {sub_cols}"
        )

    base = raw.get("baseline") or {}
    limits = raw.get("limits") or {}

    # The policy's explore branch is only reachable while explore_after < conv_n.
    # `best_history` is monotone non-decreasing, so `converged()` can only fire once
    # every one of the last conv_n iterations was flat — i.e. flat_iters >= conv_n by
    # then. If exploration needs at least that many flat iterations, the run always
    # stops on the very iteration exploration became reachable, and the branch that is
    # supposed to break a plateau never executes. Enforced here rather than left to a
    # comment: this shipped as a silent default-vs-default collision (3 and 3) and cost
    # a live run 37 unused iterations at a plateau 0.0098 under baseline.
    conv_n = int(limits.get("conv_n", 3))
    explore_after = int(limits.get("explore_after", 2))
    if explore_after < 1:
        raise TaskConfigError(
            f"{where}.limits: explore_after={explore_after}; must be >= 1"
        )
    if explore_after >= conv_n:
        raise TaskConfigError(
            f"{where}.limits: explore_after={explore_after} must be < conv_n={conv_n}, "
            "otherwise the run converges on the same iteration the policy would first "
            "explore and the explore branch can never run. Lower explore_after or raise "
            "conv_n."
        )

    return TaskConfig(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        description=description,
        data=data,
        primary_parts=primary_parts,
        report_metrics=report_metrics,
        submission_columns=sub_cols,
        prediction_column=pred_col,
        baseline_val={k: float(v) for k, v in (base.get("valid") or {}).items()},
        baseline_test={k: float(v) for k, v in (base.get("test") or {}).items()},
        ceiling=float(raw["ceiling"]) if raw.get("ceiling") is not None else None,
        max_iters=int(limits.get("max_iters", 50)),
        wall_clock_s=int(limits.get("wall_clock_s", 6 * 3600)),
        conv_eps=float(limits.get("conv_eps", 0.002)),
        conv_n=conv_n,
        explore_after=explore_after,
        seed_std=float(raw["seed_std"]) if raw.get("seed_std") is not None else None,
        ideas_path=_resolve_ideas(raw.get("ideas"), where),
        notes=str(raw.get("notes") or "").strip(),
        source_path=source,
    )


#: What a task gets when it names no bank of its own. Domain-free on purpose: inheriting
#: another dataset's idea bank means inheriting its conclusions, and the agent will dutifully
#: spend iterations on ideas that were measured false somewhere else entirely.
DEFAULT_IDEAS = TASKS_DIR / "ideas" / "generic-tabular.yaml"


def _resolve_ideas(value, where: str) -> Path | None:
    """Locate the task's idea bank; fall back to the domain-free default."""
    if not value:
        return DEFAULT_IDEAS if DEFAULT_IDEAS.is_file() else None
    p = Path(str(value))
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        raise TaskConfigError(f"{where}: ideas file not found: {p}")
    return p


def load_task(ref: str | Path) -> TaskConfig:
    """Load a task by path, or by name from `tasks/`."""
    try:
        import yaml
    except ImportError:  # pragma: no cover
        raise TaskConfigError("task files need pyyaml; add it to requirements.txt") from None

    path = Path(ref)
    if not path.suffix:
        for ext in (".yaml", ".yml"):
            cand = TASKS_DIR / f"{path.name}{ext}"
            if cand.is_file():
                path = cand
                break
    if not path.is_file() and not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        known = sorted(p.stem for p in TASKS_DIR.glob("*.y*ml")) if TASKS_DIR.is_dir() else []
        raise TaskConfigError(
            f"no task file at {ref!r}. Known tasks: {', '.join(known) or '(none)'}. "
            f"Write one in {TASKS_DIR}/<name>.yaml"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_task(raw, source=path)


def available_tasks() -> list[str]:
    if not TASKS_DIR.is_dir():
        return []
    return sorted(p.stem for p in TASKS_DIR.glob("*.y*ml"))
