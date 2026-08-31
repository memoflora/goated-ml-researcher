"""Tests for the scoring seam.

The rejection tests build tiny submissions against a synthetic 6-row split so they run
without the dataset. The reproduction tests need the real data and skip without it.

The bottom section is the end-to-end proof: two genuine reference pipelines in
`tests/fixtures/pipelines/` are executed through `sandbox.run`, their `submission.csv` is
checked by `validate` and graded by `score`, and the result is compared against the numbers
the organisers publish. Nothing else in the suite exercises that whole chain on real data,
and until it did, "the orchestrator scores a node" was an untested claim.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from orchestrator import evaluate as ev
from orchestrator.splits import DEFAULT_DATA_DIR, Split, get_split, normalise_split

HAVE_DATA = DEFAULT_DATA_DIR.is_dir()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="KuaiRand-Pure not downloaded")

SLOW_ENV = "TECHJAM_SLOW_TESTS"
slow = pytest.mark.skipif(
    os.environ.get(SLOW_ENV, "").lower() not in ("1", "true", "yes"),
    reason=f"trains a model (~2 min); set {SLOW_ENV}=1 to run",
)

PIPELINES = Path(__file__).parent / "fixtures" / "pipelines"

# Published reference numbers (vendor/starter_kit/baseline_scores.json).
BASELINE_VAL = {"gauc": 0.6674, "ndcg@5": 0.5357, "primary": 0.6016}

# The organisers' own five-seed standard deviation. A gap larger than this is a real
# disagreement with the reference implementation, not numerical drift.
SEED_STD = 0.0008


@pytest.fixture
def fake_split() -> Split:
    return Split(
        name="valid",
        user_ids=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        video_ids=np.array([10, 11, 12, 10, 13, 14], dtype=np.int64),
        labels=np.array([1, 0, 0, 0, 1, 1], dtype=np.int8),
    )


def write_csv(path: Path, rows: list[list], header: list[str] | None = None) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header if header is not None else ev.HEADER)
        w.writerows(rows)
    return path


def good_rows(split: Split) -> list[list]:
    return [
        [i, int(u), int(v), f"{0.5 - 0.1 * i:.4f}"]
        for i, (u, v) in enumerate(zip(split.user_ids, split.video_ids, strict=False))
    ]


def test_normalise_split_accepts_val_and_valid():
    assert normalise_split("val") == "valid"
    assert normalise_split("valid") == "valid"
    assert normalise_split("TEST") == "test"
    with pytest.raises(ValueError):
        normalise_split("holdout")


def test_accepts_a_well_formed_submission(tmp_path, fake_split):
    p = write_csv(tmp_path / "s.csv", good_rows(fake_split))
    ev._read_scores(p, fake_split)  # must not raise


@pytest.mark.parametrize(
    "mutate, expect",
    [
        pytest.param(lambda r: r, "", id="control"),
        pytest.param(lambda r: r[:-1], "rows", id="too-few-rows"),
        pytest.param(lambda r: r + [[6, 2, 15, "0.1"]], "more rows", id="too-many-rows"),
        pytest.param(
            lambda r: [x if i != 3 else [99, x[1], x[2], x[3]] for i, x in enumerate(r)],
            "row_id",
            id="row-id-gap",
        ),
        pytest.param(
            lambda r: [x if i != 2 else [x[0], 999, x[2], x[3]] for i, x in enumerate(r)],
            "misaligned",
            id="user-misaligned",
        ),
        pytest.param(
            lambda r: [x if i != 2 else [x[0], x[1], 999, x[3]] for i, x in enumerate(r)],
            "misaligned",
            id="video-misaligned",
        ),
        pytest.param(
            lambda r: [x if i != 1 else [x[0], x[1], x[2], "abc"] for i, x in enumerate(r)],
            "not a number",
            id="non-numeric-score",
        ),
        pytest.param(
            lambda r: [x if i != 1 else [x[0], x[1], x[2], "nan"] for i, x in enumerate(r)],
            "NaN/Inf",
            id="nan-score",
        ),
        pytest.param(
            lambda r: [x if i != 1 else [x[0], x[1], x[2], "inf"] for i, x in enumerate(r)],
            "NaN/Inf",
            id="inf-score",
        ),
        pytest.param(
            lambda r: [x if i != 1 else [x[0], x[1], x[2]] for i, x in enumerate(r)],
            "fields",
            id="wrong-field-count",
        ),
    ],
)
def test_rejection_cases(tmp_path, fake_split, mutate, expect):
    rows = mutate(good_rows(fake_split))
    p = write_csv(tmp_path / "s.csv", rows)
    if not expect:  # the control must pass
        ev._read_scores(p, fake_split)
        return
    with pytest.raises(ev.SubmissionError) as exc:
        ev._read_scores(p, fake_split)
    assert expect in str(exc.value)


def test_rejects_wrong_header(tmp_path, fake_split):
    p = write_csv(
        tmp_path / "s.csv", good_rows(fake_split), header=["row_id", "user_id", "vid", "score"]
    )
    with pytest.raises(ev.SubmissionError, match="header"):
        ev._read_scores(p, fake_split)


def test_score_arrays_matches_hand_computed(fake_split):
    """Two users, each with a perfect ranking -> GAUC 1.0, nDCG 1.0."""
    perfect = fake_split.labels.astype(float)
    m = ev.score_arrays(fake_split.user_ids, fake_split.labels, perfect)
    assert m["gauc"] == pytest.approx(1.0)
    assert m["ndcg@5"] == pytest.approx(1.0)
    assert m["primary"] == pytest.approx(1.0)


def test_score_arrays_zero_positive_user_counts_as_ndcg_zero():
    """A user with no positive contributes nDCG 0 and is excluded from GAUC."""
    users = np.array([1, 1, 2, 2], dtype=np.int64)
    labels = np.array([1, 0, 0, 0], dtype=np.int8)
    m = ev.score_arrays(users, labels, np.array([1.0, 0.0, 1.0, 0.0]))
    assert m["gauc"] == pytest.approx(1.0)  # only user 1 counts
    assert m["ndcg@5"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2


class TestRepairableMessages:
    """A rejection is read by an LLM that never sees the file. It has to carry the fix.

    Every case below asserts two things: the *diagnosis* (what is wrong, and where) and the
    *rule* (what a correct submission looks like). The second half is what an ordinary
    `raise ValueError("bad header")` leaves out, and without it the repair loop is guessing.
    """

    def _reject(self, tmp_path, split, rows, header=None) -> str:
        p = write_csv(tmp_path / "bad.csv", rows, header=header)
        with pytest.raises(ev.SubmissionError) as exc:
            ev._read_scores(p, split)
        return str(exc.value)

    def test_bad_header_names_the_offending_column(self, tmp_path, fake_split):
        msg = self._reject(
            tmp_path, fake_split, good_rows(fake_split),
            header=["row_id", "user_id", "vid", "score"],
        )
        assert "row_id,user_id,video_id,score" in msg      # what it must be
        assert "column 3 is 'vid'" in msg                  # exactly where it went wrong
        assert "'video_id'" in msg and "missing" in msg    # and which name is absent
        assert "in this order" in msg                      # and the rule

    def test_empty_file_asks_for_a_header_rather_than_reporting_None(self, tmp_path,
                                                                    fake_split):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ev.SubmissionError) as exc:
            ev._read_scores(p, fake_split)
        msg = str(exc.value)
        assert "empty" in msg and "row_id,user_id,video_id,score" in msg
        assert "None" not in msg  # the old message printed the raw `next(reader, None)`

    def test_row_count_mismatch_says_how_many_and_names_the_usual_cause(self, tmp_path,
                                                                       fake_split):
        msg = self._reject(tmp_path, fake_split, good_rows(fake_split)[:-2])
        assert "4 rows" in msg and "6" in msg and "2 missing" in msg
        assert "one row per evaluation-split row" in msg
        assert "subsample" in msg  # shrinking the submission is the classic mistake

    def test_too_many_rows_says_to_stop_not_to_pad(self, tmp_path, fake_split):
        rows = good_rows(fake_split) + [[6, 2, 15, "0.1"]]
        msg = self._reject(tmp_path, fake_split, rows)
        assert "more rows" in msg and "exactly 6 rows" in msg
        assert "header twice" in msg

    def test_row_id_gap_explains_the_indexing_rule(self, tmp_path, fake_split):
        rows = good_rows(fake_split)
        rows[3] = [99, *rows[3][1:]]
        msg = self._reject(tmp_path, fake_split, rows)
        assert "row_id=99, expected 3" in msg
        assert "0-based position" in msg
        assert "sort" in msg and "shuffle" in msg  # how a gap actually gets produced

    def test_non_integer_row_id_says_what_the_value_should_have_been(self, tmp_path,
                                                                    fake_split):
        rows = good_rows(fake_split)
        rows[2] = ["two", *rows[2][1:]]
        msg = self._reject(tmp_path, fake_split, rows)
        assert "'two' is not an integer" in msg and "0-based position" in msg

    def test_nan_score_lists_the_usual_causes(self, tmp_path, fake_split):
        rows = good_rows(fake_split)
        rows[1] = [*rows[1][:3], "nan"]
        msg = self._reject(tmp_path, fake_split, rows)
        assert "NaN/Inf" in msg and "'nan'" in msg
        assert "division by zero" in msg and "finite" in msg

    def test_misalignment_warns_that_the_pair_is_not_a_key(self, tmp_path, fake_split):
        rows = good_rows(fake_split)
        rows[2] = [rows[2][0], 999, rows[2][2], rows[2][3]]
        msg = self._reject(tmp_path, fake_split, rows)
        assert "misaligned" in msg and "(999,12)" in msg and "row 2 is (1,12)" in msg
        assert "not" in msg and "unique" in msg  # the reason a merge cannot be used

    def test_wrong_field_count_suggests_the_cause(self, tmp_path, fake_split):
        rows = good_rows(fake_split)
        rows[1] = rows[1][:3]
        msg = self._reject(tmp_path, fake_split, rows)
        assert "3 fields, expected 4" in msg and "comma" in msg

    def test_missing_file_says_where_it_should_have_been_written(self, tmp_path):
        ok, msg = ev.validate(tmp_path / "nope.csv", "val")
        assert not ok
        assert "not found" in msg and "--out-dir" in msg


def test_no_text_file_io_in_c_owned_files_relies_on_the_platform_default_encoding():
    """The same audit `test_core.py` runs over A's files, run over C's.

    Windows opens text files as cp1252 unless told otherwise; that has already produced two
    real bugs here. `_read_scores` was one of them in waiting — it parsed every submission
    with the platform default. The reference pipelines are covered too: they are the
    template the agent copies, so an implicit encoding in them propagates.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    owned = [root / "orchestrator" / f"{m}.py" for m in
             ("evaluate", "splits", "datacard", "metrics", "report", "taskspec",
              "datasource", "profile")]
    owned += [root / "tests" / f"test_{m}.py" for m in
              ("evaluator", "datacard", "report", "taskspec")]
    owned += sorted(PIPELINES.glob("kuairand_*.py"))

    offenders = []
    for path in owned:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if node.func.id != "open":
                    continue
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in {"read_text", "write_text", "open"}:
                    continue
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    continue  # os.open takes fd flags, not an encoding
            else:
                continue
            if any(kw.arg == "encoding" for kw in node.keywords):
                continue
            if any(isinstance(a, ast.Constant) and "b" in str(a.value) for a in node.args):
                continue  # binary mode
            offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, f"text I/O without an explicit encoding: {offenders}"


def test_delta_vs_baseline():
    m = {"gauc": 0.6774, "ndcg@5": 0.5457, "primary": 0.6116}
    d = ev.delta_vs_baseline(m, BASELINE_VAL)
    assert d["primary"] == pytest.approx(0.0100, abs=1e-6)
    assert d["gauc"] == pytest.approx(0.0100, abs=1e-6)


# --------------------------------------------------------------------------- with real data


@needs_data
def test_validate_reports_missing_file():
    ok, msg = ev.validate(Path("/nonexistent/submission.csv"), "val")
    assert not ok and "not found" in msg


@needs_data
def test_hidden_test_scoring_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(ev.TEST_SCORING_ENV, raising=False)
    # "held-out" rather than "hidden": the guard now covers any task, and only KuaiRand's
    # test split is hidden by an organiser. The refusal itself is unchanged.
    with pytest.raises(ev.SubmissionError, match="refusing to score the held-out test"):
        ev.score(tmp_path / "anything.csv", "test")


@needs_data
def test_split_sizes_match_the_official_numbers():
    from orchestrator.splits import load_splits

    sizes = {k: len(v) for k, v in load_splits().items()}
    assert sizes == {"train": 1_141_112, "valid": 124_909, "test": 170_588}


@needs_data
def test_item_popularity_rung_reproduces():
    """The published sanity rung: item popularity -> validation primary 0.5807.

    If this drifts, the harness is wrong and every metric we report is wrong with it.
    """
    import collections

    from orchestrator.splits import get_split

    tr, va = get_split("train"), get_split("valid")
    pos, imp = collections.Counter(), collections.Counter()
    for v, y in zip(tr.video_ids.tolist(), tr.labels.tolist(), strict=False):
        imp[v] += 1
        pos[v] += y
    gmean = sum(pos.values()) / sum(imp.values())
    prior = 20.0
    scores = np.array(
        [
            (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
            for v in va.video_ids.tolist()
        ]
    )
    m = ev.score_arrays(va.user_ids, va.labels, scores)
    assert m["primary"] == pytest.approx(0.5807, abs=0.0005)
    assert m["gauc"] == pytest.approx(0.6387, abs=0.0005)
    assert m["ndcg@5"] == pytest.approx(0.5227, abs=0.0005)


# ------------------------------------------------- the whole seam, on the real dataset
#
# `tests/fixtures/pipelines/kuairand_pop.py` and `kuairand_fm.py` are not fault fixtures.
# They are genuine, self-contained solutions that honour the frozen pipeline CLI, and they
# exist so that this file can answer the one question nothing else answers:
#
#     does  sandbox.run -> evaluate.validate -> evaluate.score  produce the right number
#     on the real data, end to end, with no orchestrator internals short-circuited?
#
# Every constant below was measured through that chain, not copied from the organisers'
# README, and the comment on each says how it relates to the published figure.

#: pop, `--split val`. Identical to the organisers' `baseline.py --model pop` to 4 dp.
POP_VAL = {"gauc": 0.6387, "ndcg@5": 0.5227, "primary": 0.5807}

#: pop, `--split test`. HIGHER than the organisers' published 0.5715 *by construction*:
#: contracts.md §1.2 makes `--split test` fit on train + validation, while `baseline.py`
#: reports a train-only fit on both splits. The extra week is worth +0.0034.
POP_TEST_PRIMARY = 0.5749
POP_TEST_TRAIN_ONLY_PRIMARY = 0.5715

#: FM, `--split val`. The organisers publish 0.6016; running their own untouched
#: `baseline.py` loop on this machine also gives 0.60147, so the 0.00013 gap is their
#: environment, not our harness — well inside the five-seed std of 0.0008.
FM_VAL_PRIMARY = 0.6015


def run_pipeline(tmp_path: Path, fixture: str, split: str, *, seed: int = 0,
                 timeout_s: int = 900, subsample: float | None = None):
    """Execute a fixture pipeline exactly the way the orchestrator executes a node."""
    from orchestrator import sandbox
    from orchestrator.contracts import Node

    ws = tmp_path / "node"
    ws.mkdir(parents=True, exist_ok=True)
    shutil.copy(PIPELINES / fixture, ws / "pipeline.py")
    node = Node(id="n000", parent_id=None, kind="draft", iteration=0, workspace=ws)
    res = sandbox.run(node, split=split, seed=seed, timeout_s=timeout_s,
                      subsample=subsample, data_dir=DEFAULT_DATA_DIR)
    assert res.ok, (
        f"{fixture} --split {split} failed: {res.error_class}: {res.error_excerpt}\n"
        f"stderr: {res.stderr_tail}"
    )
    return res


def load_fixture_module(name: str):
    """Import a fixture pipeline by path — they are deliberately not importable packages."""
    spec = importlib.util.spec_from_file_location(f"_fixture_{name}", PIPELINES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_data
def test_pop_pipeline_scores_the_published_validation_numbers(tmp_path):
    """The end-to-end proof. Run the pipeline, validate the CSV, score it, check the number.

    This is the test that fails if any link in the chain breaks: the sandbox's argv, the
    submission format, the row ordering, the split boundaries, or the metric delegation.
    """
    res = run_pipeline(tmp_path, "kuairand_pop.py", "val")
    sub = res.artifacts["submission"]

    assert res.result_json["n_rows"] == 124_909
    ok, msg = ev.validate(sub, "val")
    assert ok, msg
    assert "124,909 rows" in msg and "split=valid" in msg

    m = ev.score(sub, "val")
    for key, want in POP_VAL.items():
        assert m[key] == pytest.approx(want, abs=SEED_STD), f"{key}: {m}"


@needs_data
def test_pop_pipeline_on_test_split_fits_on_train_plus_validation(tmp_path, monkeypatch):
    """`--split test` must use the extra week, and the score must show that it did.

    Scoring test is the guarded path, so this is also the only test that proves the guard
    can be opened deliberately without weakening it — the refusal is asserted elsewhere.
    """
    res = run_pipeline(tmp_path, "kuairand_pop.py", "test")
    sub = res.artifacts["submission"]

    assert res.result_json["n_rows"] == 170_588
    assert "train+valid" in res.result_json["notes"]
    ok, msg = ev.validate(sub, "test")  # validation needs no labels and no env var
    assert ok, msg

    monkeypatch.setenv(ev.TEST_SCORING_ENV, "1")
    m = ev.score(sub, "test")
    assert m["primary"] == pytest.approx(POP_TEST_PRIMARY, abs=SEED_STD), m
    assert m["primary"] > POP_TEST_TRAIN_ONLY_PRIMARY, (
        "fitting on train+valid should beat the published train-only test score"
    )


@needs_data
def test_subsample_shrinks_training_and_never_the_submission(tmp_path):
    """The contract's easiest trap: `--subsample` samples users to train on, not rows to
    emit. A pipeline that shrinks the submission instead writes a file that can never be
    scored, so the fixture has to get this right or it is teaching the wrong shape."""
    res = run_pipeline(tmp_path, "kuairand_pop.py", "val", subsample=0.02)
    assert res.result_json["n_rows"] == 124_909
    ok, msg = ev.validate(res.artifacts["submission"], "val")
    assert ok, msg


@needs_data
def test_pop_fixture_row_order_matches_the_starter_kit_loader():
    """The fixture re-derives the splits instead of importing them — the contract forbids a
    pipeline from importing our code. That duplication is the risk this test exists to
    remove: if the fixture's date filter, file order or label rule ever drifts from
    `vendor/starter_kit/data.py`, the submission silently misaligns and every score after
    it is meaningless."""
    pop = load_fixture_module("kuairand_pop")
    df = pop.load_logs(pop.resolve_data_dir(str(DEFAULT_DATA_DIR)))

    for name, n_rows in (("train", 1_141_112), ("valid", 124_909), ("test", 170_588)):
        got = pop.slice_split(df, name)
        want = get_split(name)
        assert len(got) == n_rows == len(want)
        assert np.array_equal(got["user_id"].to_numpy(), want.user_ids), name
        assert np.array_equal(got["video_id"].to_numpy(), want.video_ids), name
        assert np.array_equal(got["y"].to_numpy().astype(np.int8), want.labels), name


@needs_data
@slow
def test_fm_pipeline_reproduces_the_official_baseline(tmp_path):
    """The second reference: a *trained* model, not a statistic, through the same seam.

    Slow (~2 min), so it is opt-in. It is the only check that the seam survives a pipeline
    with an encoder, minibatch training and early stopping — the shape of everything the
    agent will actually write.
    """
    res = run_pipeline(tmp_path, "kuairand_fm.py", "val", timeout_s=1800)
    sub = res.artifacts["submission"]

    assert res.result_json["n_rows"] == 124_909
    ok, msg = ev.validate(sub, "val")
    assert ok, msg

    m = ev.score(sub, "val")
    assert m["primary"] == pytest.approx(FM_VAL_PRIMARY, abs=SEED_STD), m
    assert m["primary"] == pytest.approx(BASELINE_VAL["primary"], abs=SEED_STD), m
    assert m["primary"] > POP_VAL["primary"], "FM must beat item popularity"


@needs_data
@slow
def test_fm_fixture_encoder_matches_the_starter_kit_encoder():
    """The FM fixture vectorises `data.py::encode`. Vectorising it replaced the Python dict
    vocabulary with `pd.factorize`; if those ever assign ids in a different order the model
    is still trainable but is no longer the published baseline. Assert the feature matrices
    are identical, cell for cell."""
    import sys

    fm = load_fixture_module("kuairand_fm")
    sys.path.insert(0, str(ev.STARTER_KIT))
    try:
        import data as starter_data  # type: ignore[import-not-found]

        raw = starter_data.load(str(DEFAULT_DATA_DIR))
        enc_ref, dim_ref = starter_data.encode(raw)
    finally:
        sys.path.remove(str(ev.STARTER_KIT))
        sys.modules.pop("data", None)

    df = fm.load_logs(fm.resolve_data_dir(str(DEFAULT_DATA_DIR)))
    train, valid = fm.slice_split(df, "train"), fm.slice_split(df, "valid")
    enc_got, dim_got = fm.encode(train, {"valid": valid})

    assert dim_got == dim_ref
    assert np.array_equal(enc_got["__fit__"][0], enc_ref["train"][0])
    assert np.array_equal(enc_got["__fit__"][1], enc_ref["train"][1])
    assert np.array_equal(enc_got["valid"][0], enc_ref["valid"][0])


class TestPipelineWhitelistIsHonest:
    """The whitelist is a promise to the agent, and the sandbox runs pipelines on this
    same interpreter. A library listed but not importable is the worst kind of entry:
    the agent is invited to use it, writes a pipeline around it, and the run burns an
    iteration on a guaranteed ModuleNotFoundError. It happened on the first live run —
    the model reached for `torch`, which was listed and absent.
    """

    @staticmethod
    def _entries():
        import re
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "requirements-pipeline.txt"
        for raw in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", raw.strip())
            if m:
                yield m.group(1), m.group(2)

    def test_every_listed_library_is_importable(self):
        import importlib.util

        aliases = {"scikit-learn": "sklearn", "pyyaml": "yaml"}
        missing = [
            name
            for name, _ in self._entries()
            if importlib.util.find_spec(aliases.get(name, name.replace("-", "_"))) is None
        ]
        assert not missing, (
            f"requirements-pipeline.txt offers {missing} to the agent, but they cannot "
            f"be imported by the interpreter the sandbox runs pipelines with. Either "
            f"install them or remove them from the whitelist — a whitelist that lies "
            f"costs an iteration every time the agent believes it."
        )

    def test_pins_match_what_is_installed(self):
        import importlib.metadata as md

        drift = []
        for name, pinned in self._entries():
            try:
                actual = md.version(name)
            except md.PackageNotFoundError:
                continue  # covered by the test above
            if actual != pinned:
                drift.append(f"{name}: pinned {pinned}, installed {actual}")
        assert not drift, (
            "the whitelist pins versions the environment does not have: "
            + "; ".join(drift)
        )
