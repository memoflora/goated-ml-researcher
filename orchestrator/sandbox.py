"""Run an agent-written `pipeline.py` in isolation and classify how it failed.

OWNER: B (ML Engineer — Agent Runtime & Sandbox).

Contract (references/contracts.md §1):

    python pipeline.py --data-dir DIR --out-dir DIR --split {val,test} --seed N [--subsample F]

The pipeline must write `<out-dir>/submission.csv` and print exactly one stdout line
`RESULT_JSON {...}`. Scoring never happens here — the orchestrator scores the CSV with
C's `evaluate.py`. This module only decides *whether it ran* and, if not, *why*, in the
smallest number of tokens that lets the LLM fix it.

Design rules:
  * Fail loudly inside, recover outside. We never repair code here.
  * Kill the whole process group, never just the child.
  * No network during training — a pipeline that downloads data would breach the
    no-external-data rule, which is the one disqualifying rule.
  * No secret ever reaches the child environment or the node workspace.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .contracts import ErrorClass, ExecResult, Node

DEFAULT_TIMEOUT_S = 25 * 60
DEFAULT_MEM_LIMIT_MB = 8192
TAIL_CHARS = 4000
EXCERPT_CHARS = 1500
RESULT_PREFIX = "RESULT_JSON"
SUBMISSION_HEADER = "row_id,user_id,video_id,score"
TERM_GRACE_S = 3.0
_TIME_POLL_S = 0.2
_RSS_POLL_S = 1.0

# Anything matching these never reaches the child process or the workspace.
_SECRET_PATTERNS = (
    re.compile(r"API_KEY$"), re.compile(r"^ANTHROPIC_"), re.compile(r"TOKEN$"),
    re.compile(r"SECRET"), re.compile(r"PASSWORD"), re.compile(r"^AWS_"),
    re.compile(r"^OPENAI_"), re.compile(r"^GH_"), re.compile(r"^GITHUB_TOKEN"),
)

# Injected as sitecustomize.py in the node workspace. Blocks outbound sockets so a
# generated pipeline cannot download data, while leaving loopback alone (some
# libraries open local sockets for multiprocessing).
_NETWORK_GUARD = '''\
"""Injected by orchestrator/sandbox.py. Blocks non-loopback network access."""
import socket as _s

_ALLOWED = ("127.0.0.1", "::1", "localhost", "0.0.0.0")


class NetworkBlocked(OSError):
    pass


def _host_of(address):
    if isinstance(address, (tuple, list)) and address:
        return str(address[0])
    return str(address)


def _guard(name, original):
    def wrapper(self, address, *a, **kw):
        if _host_of(address) not in _ALLOWED:
            raise NetworkBlocked(
                "network access is disabled inside the sandbox (%s to %r). "
                "Use only the local files under --data-dir; downloading any "
                "external data is forbidden." % (name, address)
            )
        return original(self, address, *a, **kw)
    return wrapper


_s.socket.connect = _guard("connect", _s.socket.connect)
_s.socket.connect_ex = _guard("connect_ex", _s.socket.connect_ex)
'''


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def run(node: Node, *, split: str, seed: int,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        subsample: float | None = None,
        data_dir: Path | None = None,
        mem_limit_mb: int = DEFAULT_MEM_LIMIT_MB,
        python: str | None = None,
        allow_network: bool = False) -> ExecResult:
    """Execute `node.workspace/pipeline.py` and return a classified ExecResult.

    Never raises for a misbehaving pipeline: every failure mode comes back as an
    ExecResult with `ok=False` and an `error_class`. Raises only if the workspace
    itself is unusable, which is an orchestrator bug, not an agent one.
    """
    ws = Path(node.workspace)
    script = ws / "pipeline.py"
    if not script.is_file():
        raise FileNotFoundError(f"no pipeline.py in node workspace {ws}")

    data_dir = Path(data_dir or os.environ.get("TECHJAM_DATA_DIR", "data")).resolve()
    _clean_artifacts(ws)

    cmd = [python or sys.executable, "-u", "pipeline.py",
           "--data-dir", str(data_dir), "--out-dir", ".",
           "--split", split, "--seed", str(seed)]
    if subsample is not None:
        cmd += ["--subsample", str(subsample)]

    if not allow_network:
        (ws / "sitecustomize.py").write_text(_NETWORK_GUARD)

    out_path, err_path = ws / "stdout.log", ws / "stderr.log"
    started = time.monotonic()
    with open(out_path, "wb") as out_fh, open(err_path, "wb") as err_fh:
        proc = subprocess.Popen(
            cmd, cwd=str(ws), stdout=out_fh, stderr=err_fh,
            stdin=subprocess.DEVNULL,           # never prompt for input
            env=_child_env(ws, seed, allow_network),
            start_new_session=True,             # its own process group, so we can killpg
            preexec_fn=_rlimits(mem_limit_mb),  # noqa: PLW1509 - posix only, by design
        )
        kill_reason, peak_rss_mb = _supervise(proc, timeout_s, mem_limit_mb, started)
    wall_s = time.monotonic() - started

    stdout = _read_text(out_path)
    stderr = _read_text(err_path)
    return _build_result(ws, proc.returncode, stdout, stderr,
                         kill_reason, wall_s, peak_rss_mb)


def classify(exit_code: int, stderr: str, stdout: str = "",
             kill_reason: str | None = None) -> ErrorClass:
    """Map a failed execution onto an ErrorClass. Pure, so it unit-tests cheaply."""
    if kill_reason == "timeout":
        return "timeout"
    if kill_reason == "memory":
        return "oom"

    blob = f"{stderr}\n{stdout}"
    # The child dying on SIGKILL with no traceback is the OS OOM killer.
    if exit_code in (-signal.SIGKILL, 137) and not _has_traceback(stderr):
        return "oom"
    if re.search(r"\bMemoryError\b|Cannot allocate memory|std::bad_alloc|"
                 r"Unable to allocate .* for an array|numpy\.core\._exceptions\._ArrayMemoryError",
                 blob):
        return "oom"
    if re.search(r"^\s*(SyntaxError|IndentationError|TabError):", blob, re.M):
        return "syntax"
    if re.search(r"^\s*(ModuleNotFoundError|ImportError):", blob, re.M):
        return "import"
    if "NetworkBlocked" in blob:
        return "data"
    if re.search(r"^\s*FileNotFoundError:", blob, re.M) or re.search(
            r"^\s*(pandas\.errors\.\w*ParserError|UnicodeDecodeError|csv\.Error):",
            blob, re.M):
        return "data"
    if re.search(r"nan|inf(inity)?", blob, re.I) and re.search(r"score", blob, re.I):
        return "eval"
    if _has_traceback(blob) or exit_code != 0:
        return "runtime"
    return "unknown"


def excerpt(stderr: str, *, limit: int = EXCERPT_CHARS,
            script_name: str = "pipeline.py") -> str | None:
    """The single most useful slice of a traceback: the deepest `pipeline.py` frame
    plus the exception line. Feeding the LLM 200 lines of traceback is how token
    budgets die."""
    if not stderr.strip():
        return None
    block = _last_traceback_block(stderr)
    if block is None:
        tail = stderr.strip().splitlines()[-8:]
        return _cap("\n".join(tail), limit)

    lines = block.splitlines()
    frames = [i for i, ln in enumerate(lines)
              if ln.lstrip().startswith("File ") and script_name in ln]
    exc_idx = _exception_line_index(lines)

    picked: list[str] = []
    if frames:
        picked = _one_frame(lines, frames[-1])    # File..., source line, caret
    elif len(lines) > 1:
        picked = _one_frame(lines, 1)
    if exc_idx is not None and (not picked or lines[exc_idx] not in picked):
        picked.append(lines[exc_idx])
    return _cap("\n".join(ln.rstrip() for ln in picked if ln.strip()), limit)


def parse_result_json(stdout: str) -> tuple[dict | None, str | None]:
    """Return (payload, error) for the RESULT_JSON line. Last one wins if repeated."""
    candidates = [ln for ln in stdout.splitlines() if ln.lstrip().startswith(RESULT_PREFIX)]
    if not candidates:
        return None, f"no `{RESULT_PREFIX} {{...}}` line on stdout"
    raw = candidates[-1].lstrip()[len(RESULT_PREFIX):].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"`{RESULT_PREFIX}` line is not valid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return None, f"`{RESULT_PREFIX}` payload is {type(payload).__name__}, expected object"
    missing = [k for k in ("n_rows", "train_seconds", "notes") if k not in payload]
    if missing:
        return payload, f"`{RESULT_PREFIX}` is missing required key(s): {', '.join(missing)}"
    return payload, None


def check_submission(path: Path, *, expected_rows: int | None = None
                     ) -> tuple[ErrorClass | None, str | None]:
    """Cheap structural guard so a malformed CSV is caught before C's evaluator.

    This is not scoring and not `evaluate.validate()` — it only catches the shapes
    the fault-injection suite must recover from: no file, wrong header, NaN/Inf
    scores, and a row count that contradicts the pipeline's own RESULT_JSON.
    """
    if not path.is_file():
        return "contract", "pipeline exited 0 but wrote no submission.csv"
    with open(path, newline="") as fh:
        header = fh.readline().strip().lstrip("﻿")
        if header != SUBMISSION_HEADER:
            return "contract", f"submission header is {header!r}, expected {SUBMISSION_HEADER!r}"
        n_rows = 0
        for lineno, line in enumerate(fh, start=2):
            if not line.strip():
                continue
            n_rows += 1
            parts = line.rstrip("\r\n").split(",")
            if len(parts) != 4:
                return "contract", f"submission line {lineno} has {len(parts)} fields, expected 4"
            try:
                value = float(parts[3])
            except ValueError:
                return "eval", f"submission line {lineno} has non-numeric score {parts[3]!r}"
            if value != value or value in (float("inf"), float("-inf")):
                return "eval", f"submission line {lineno} has a NaN/Inf score ({parts[3]!r})"
    if n_rows == 0:
        return "contract", "submission.csv has a header but no rows"
    if expected_rows is not None and n_rows != expected_rows:
        return ("contract",
                f"submission has {n_rows} rows but RESULT_JSON reported n_rows={expected_rows}")
    return None, None


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #

def _build_result(ws: Path, returncode: int, stdout: str, stderr: str,
                  kill_reason: str | None, wall_s: float, peak_rss_mb: float) -> ExecResult:
    submission = ws / "submission.csv"
    payload, payload_err = parse_result_json(stdout)

    def fail(cls: ErrorClass, why: str) -> ExecResult:
        return ExecResult(
            ok=False, exit_code=returncode,
            stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
            error_class=cls, error_excerpt=_cap(why, EXCERPT_CHARS),
            result_json=payload, artifacts={}, wall_s=wall_s, peak_rss_mb=peak_rss_mb,
        )

    if kill_reason == "timeout":
        return fail("timeout", f"killed after exceeding the {wall_s:.0f}s time limit; "
                               "the pipeline must finish well inside its budget")
    if kill_reason == "memory":
        return fail("oom", f"killed after exceeding the memory limit "
                           f"(peak RSS {peak_rss_mb:.0f} MB)")
    if returncode != 0:
        cls = classify(returncode, stderr, stdout, kill_reason)
        return fail(cls, excerpt(stderr) or f"exited with status {returncode}")

    # exit 0 from here on: everything left is a contract or eval breach.
    if payload_err:
        return fail("contract", payload_err)
    cls, why = check_submission(submission, expected_rows=_expected_rows(payload))
    if cls:
        return fail(cls, why or "malformed submission.csv")

    return ExecResult(
        ok=True, exit_code=0,
        stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
        error_class=None, error_excerpt=None,
        result_json=payload, artifacts={"submission": submission},
        wall_s=wall_s, peak_rss_mb=peak_rss_mb,
    )


def _supervise(proc: subprocess.Popen, timeout_s: float, mem_limit_mb: int,
               started: float) -> tuple[str | None, float]:
    """Poll the child; kill its whole process group on timeout or memory blow-up."""
    peak = 0.0
    next_rss_poll = 0.0
    while True:
        if proc.poll() is not None:
            return None, peak
        elapsed = time.monotonic() - started
        if elapsed >= next_rss_poll:
            next_rss_poll = elapsed + _RSS_POLL_S
            rss = _group_rss_mb(proc.pid)
            peak = max(peak, rss)
            if mem_limit_mb and rss > mem_limit_mb:
                _kill_group(proc)
                return "memory", peak
        if elapsed >= timeout_s:
            _kill_group(proc)
            return "timeout", peak
        time.sleep(_TIME_POLL_S)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the group, then SIGKILL it. Leaving orphans behind is a failure mode
    of its own: they eat the machine for the rest of a six-hour run."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, grace in ((signal.SIGTERM, TERM_GRACE_S), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            break
        except subprocess.TimeoutExpired:
            continue
    # Reap anything in the group the child itself spawned and orphaned.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _group_rss_mb(pid: int) -> float:
    """Summed RSS of the child's process group, in MB. `ps` keeps this dependency
    free; it is polled once a second, so the cost is irrelevant."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return 0.0
    ps = shutil.which("ps")
    if not ps:
        return 0.0
    try:
        out = subprocess.run([ps, "-A", "-o", "pgid=,rss="],
                             capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return 0.0
    total_kb = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit() \
                and int(parts[0]) == pgid:
            total_kb += int(parts[1])
    return total_kb / 1024.0


def _rlimits(mem_limit_mb: int):
    """Hard backstop under our own poller: the child gets MemoryError rather than
    swapping the machine to death between two 1s samples."""
    if not mem_limit_mb:
        return None

    def apply() -> None:                                    # pragma: no cover - child
        cap = int(mem_limit_mb * 1.25) * 1024 * 1024
        for which in (resource.RLIMIT_AS, getattr(resource, "RLIMIT_DATA", None)):
            if which is None:
                continue
            try:
                soft, hard = resource.getrlimit(which)
                limit = cap if hard in (resource.RLIM_INFINITY, -1) else min(cap, hard)
                resource.setrlimit(which, (limit, hard))
            except (ValueError, OSError):
                pass
    return apply


def _child_env(ws: Path, seed: int, allow_network: bool) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _is_secret(k)}
    env["PYTHONHASHSEED"] = str(seed)               # reruns must reproduce
    env["MPLBACKEND"] = "Agg"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.pop("PYTHONSTARTUP", None)
    if not allow_network:
        # sitecustomize.py lives in the workspace; make sure it is importable and
        # that nothing upstream has disabled site processing.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ws)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["no_proxy"] = "*"
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(var, None)
    return env


def _is_secret(name: str) -> bool:
    return any(p.search(name) for p in _SECRET_PATTERNS)


def _clean_artifacts(ws: Path) -> None:
    """A rerun must not inherit the previous attempt's submission — that is how a
    broken node silently scores like its parent."""
    for name in ("submission.csv", "stdout.log", "stderr.log"):
        (ws / name).unlink(missing_ok=True)


def _expected_rows(payload: dict | None) -> int | None:
    if not payload:
        return None
    value = payload.get("n_rows")
    return value if isinstance(value, int) and value >= 0 else None


def _read_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _tail(text: str, limit: int = TAIL_CHARS) -> str:
    return text if len(text) <= limit else text[-limit:]


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _has_traceback(text: str) -> bool:
    return "Traceback (most recent call last)" in text


def _last_traceback_block(stderr: str) -> str | None:
    idx = stderr.rfind("Traceback (most recent call last)")
    return stderr[idx:] if idx != -1 else None


def _one_frame(lines: list[str], start: int, max_lines: int = 3) -> list[str]:
    """One traceback frame: its `File ...` line and the source lines under it,
    stopping before the next frame."""
    out = [lines[start]]
    for ln in lines[start + 1:start + max_lines]:
        if ln.lstrip().startswith("File "):
            break
        out.append(ln)
    return out


def _exception_line_index(lines: list[str]) -> int | None:
    """Last line that looks like `SomeError: message` at column zero."""
    pattern = re.compile(r"^(\w[\w.]*(Error|Exception|Exit|Interrupt|Warning))\b")
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i]
        if stripped and not stripped[0].isspace() and pattern.match(stripped):
            return i
    return None
