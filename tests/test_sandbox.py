"""Unit tests for the pure parts of the sandbox: classification, excerpting,
RESULT_JSON parsing and the structural submission guard."""
from __future__ import annotations

import signal

import pytest

from orchestrator.sandbox import (
    SUBMISSION_HEADER,
    check_submission,
    classify,
    excerpt,
    native_crash_note,
    parse_result_json,
)

TB = """Traceback (most recent call last):
  File "pipeline.py", line 120, in <module>
    main()
  File "pipeline.py", line 88, in main
    scores = rank(model, val)
  File "/usr/lib/python3.11/site-packages/numpy/lib/x.py", line 12, in inner
    return arr[idx]
IndexError: index 5 is out of bounds for axis 0 with size 3
"""


class TestClassify:
    @pytest.mark.parametrize("stderr,expected", [
        ('  File "pipeline.py", line 3\n    a = (\nSyntaxError: invalid syntax', "syntax"),
        ("IndentationError: unexpected indent", "syntax"),
        ("ModuleNotFoundError: No module named 'torch'", "import"),
        ("ImportError: cannot import name 'foo'", "import"),
        ("MemoryError", "oom"),
        ("numpy.core._exceptions._ArrayMemoryError: Unable to allocate 8 GiB", "oom"),
        ("FileNotFoundError: [Errno 2] No such file: 'log.csv'", "data"),
        ("NetworkBlocked: network access is disabled inside the sandbox", "data"),
        (TB, "runtime"),
    ])
    def test_from_stderr(self, stderr, expected):
        assert classify(1, stderr) == expected

    def test_kill_reason_wins_over_stderr(self):
        assert classify(-9, "MemoryError", kill_reason="timeout") == "timeout"
        assert classify(-9, "", kill_reason="memory") == "oom"

    def test_sigkill_without_traceback_reads_as_oom(self):
        if not hasattr(signal, "SIGKILL"):
            pytest.skip("SIGKILL is POSIX-only; Windows never delivers it")
        assert classify(-signal.SIGKILL, "") == "oom"
        assert classify(137, "") == "oom"

    def test_traceback_beats_bare_sigkill(self):
        if not hasattr(signal, "SIGKILL"):
            pytest.skip("SIGKILL is POSIX-only; Windows never delivers it")
        assert classify(-signal.SIGKILL, TB) == "runtime"

    def test_a_fatal_signal_without_a_traceback_is_a_native_crash(self):
        """Not `runtime`: there is no traceback, so a repair has nothing to read.

        Run r20260831-0741 lost its submission here — three repair attempts against
        a bare "exited with status -11" produced the same crash three times.
        """
        for name in ("SIGSEGV", "SIGABRT", "SIGILL", "SIGFPE"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            assert classify(-int(sig), "") == "native_crash", name
            assert classify(128 + int(sig), "") == "native_crash", f"128+{name}"

    def test_windows_access_violation_is_a_native_crash(self):
        """0xC0000005 is what the winning node actually died with, on Windows."""
        assert classify(0xC0000005, "") == "native_crash"
        assert classify(0xC00000FD, "") == "native_crash"

    def test_a_traceback_beats_a_fatal_signal(self):
        """A Python exception is readable, so it stays `runtime` and keeps the excerpt."""
        assert classify(-int(signal.SIGSEGV), TB) == "runtime"

    def test_sigkill_is_still_oom_not_a_native_crash(self):
        """The OOM killer must not be swallowed by the new class."""
        if not hasattr(signal, "SIGKILL"):
            pytest.skip("SIGKILL is POSIX-only")
        assert classify(-signal.SIGKILL, "") == "oom"

    def test_bad_alloc_is_oom_even_though_it_aborts(self):
        """C++ allocation failure aborts, but it is a memory problem, not a crash."""
        assert classify(-int(signal.SIGABRT), "terminate called: std::bad_alloc") == "oom"

    def test_kill_reason_still_wins_over_a_fatal_signal(self):
        assert classify(-int(signal.SIGSEGV), "", kill_reason="timeout") == "timeout"
        assert classify(-int(signal.SIGSEGV), "", kill_reason="memory") == "oom"

    def test_the_note_names_the_cause_and_the_levers(self):
        """`excerpt()` returns None on a crash, so this text is all the model gets."""
        note = native_crash_note(-int(signal.SIGSEGV), 7900.0)
        assert "SIGSEGV" in note
        assert "7900 MB" in note
        for lever in ("no Python traceback", "same code will crash", "configuration"):
            assert lever in note, lever
        assert "0xC0000005" in native_crash_note(0xC0000005, 12.0)

    def test_clean_exit_is_unknown_not_runtime(self):
        assert classify(0, "") == "unknown"


class TestExcerpt:
    def test_picks_deepest_pipeline_frame_and_exception(self):
        got = excerpt(TB)
        assert "line 88, in main" in got
        assert "IndexError: index 5 is out of bounds" in got
        assert "line 120" not in got          # shallower frame dropped
        assert "numpy/lib/x.py" not in got    # library frame dropped

    def test_capped(self):
        got = excerpt("Traceback (most recent call last):\n" + "x" * 9000, limit=200)
        assert len(got) <= 200

    def test_no_traceback_falls_back_to_tail(self):
        got = excerpt("line one\nline two\nfatal: something broke")
        assert "fatal: something broke" in got

    def test_empty(self):
        assert excerpt("   \n") is None

    def test_syntax_error_has_no_pipeline_frame_but_still_reports(self):
        raw = ('  File "pipeline.py", line 6\n    a = (\n        ^\n'
               "SyntaxError: '(' was never closed\n")
        got = excerpt(raw)
        assert "SyntaxError" in got


class TestParseResultJson:
    def test_happy(self):
        payload, err = parse_result_json(
            'noise\nRESULT_JSON {"n_rows": 3, "train_seconds": 1.5, "notes": "fm"}\n')
        assert err is None and payload["n_rows"] == 3

    def test_missing(self):
        payload, err = parse_result_json("no marker here")
        assert payload is None and "no `RESULT_JSON" in err

    def test_malformed(self):
        payload, err = parse_result_json("RESULT_JSON {not json}")
        assert payload is None and "not valid JSON" in err

    def test_missing_required_keys(self):
        payload, err = parse_result_json('RESULT_JSON {"n_rows": 3}')
        assert payload == {"n_rows": 3}
        assert "train_seconds" in err and "notes" in err

    def test_last_line_wins(self):
        payload, _ = parse_result_json(
            'RESULT_JSON {"n_rows": 1, "train_seconds": 0, "notes": "a"}\n'
            'RESULT_JSON {"n_rows": 2, "train_seconds": 0, "notes": "b"}')
        assert payload["n_rows"] == 2

    def test_non_object_payload(self):
        payload, err = parse_result_json("RESULT_JSON [1, 2]")
        assert payload is None and "expected object" in err


class TestCheckSubmission:
    def write(self, tmp_path, text):
        p = tmp_path / "submission.csv"
        p.write_text(text, encoding="utf-8")
        return p

    def test_valid(self, tmp_path):
        p = self.write(tmp_path, f"{SUBMISSION_HEADER}\n0,u1,v1,0.5\n1,u1,v2,0.25\n")
        assert check_submission(p, expected_rows=2) == (None, None)

    def test_missing_file(self, tmp_path):
        cls, why = check_submission(tmp_path / "nope.csv")
        assert cls == "contract" and "no submission.csv" in why

    def test_wrong_header(self, tmp_path):
        p = self.write(tmp_path, "user_id,video_id,score\n1,2,0.5\n")
        assert check_submission(p)[0] == "contract"

    def test_nan_score(self, tmp_path):
        p = self.write(tmp_path, f"{SUBMISSION_HEADER}\n0,u1,v1,nan\n")
        cls, why = check_submission(p)
        assert cls == "eval" and "NaN/Inf" in why

    def test_inf_score(self, tmp_path):
        p = self.write(tmp_path, f"{SUBMISSION_HEADER}\n0,u1,v1,-Infinity\n")
        assert check_submission(p)[0] == "eval"

    def test_non_numeric_score(self, tmp_path):
        p = self.write(tmp_path, f"{SUBMISSION_HEADER}\n0,u1,v1,high\n")
        assert check_submission(p)[0] == "eval"

    def test_row_count_disagrees_with_result_json(self, tmp_path):
        p = self.write(tmp_path, f"{SUBMISSION_HEADER}\n0,u1,v1,0.5\n")
        cls, why = check_submission(p, expected_rows=9)
        assert cls == "contract" and "RESULT_JSON reported n_rows=9" in why

    def test_header_only(self, tmp_path):
        p = self.write(tmp_path, f"{SUBMISSION_HEADER}\n")
        assert check_submission(p)[0] == "contract"

    def test_wrong_field_count(self, tmp_path):
        p = self.write(tmp_path, f"{SUBMISSION_HEADER}\n0,u1,0.5\n")
        assert check_submission(p)[0] == "contract"

    def test_tolerates_bom_and_trailing_blank_line(self, tmp_path):
        p = self.write(tmp_path, f"﻿{SUBMISSION_HEADER}\n0,u1,v1,0.5\n\n")
        assert check_submission(p, expected_rows=1) == (None, None)
