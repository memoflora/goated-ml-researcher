"""Append-only JSONL journal + run accounting. Owner: A.

`runs/<run_id>/journal.jsonl` is a **graded deliverable**, not a log file. Judges
read it to score Autonomy, Robustness and Innovation. Therefore:

* one valid single-line JSON object per line, always;
* flushed and fsync-batched on every write, so a `kill -9` loses nothing;
* never a secret, never a raw traceback longer than the cap.

Public API is `emit(event: dict) -> None` (contracts.md §3).
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # `Self` is 3.11+; never evaluated at runtime, so 3.10 still works
    from typing import Self

#: Field order in every written line. Readers do not care; humans diffing do.
_FIELD_ORDER = (
    "ts",
    "run_id",
    "iteration",
    "node_id",
    "parent_id",
    "event",
    "kind",
    "hypothesis",
    "plan",
    "idea_ids",
    "metrics",
    "delta_vs_baseline",
    "error_class",
    "recovery",
    "tokens_in",
    "tokens_out",
    "model",
    "wall_s",
)

_MAX_STR = 4000  # hard cap on any single string value written to the journal

# Anything that looks like an API key never reaches disk. Belt and braces: we
# also redact the live value of ANTHROPIC_API_KEY if it is set.
_KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")
_REDACTED = "[REDACTED]"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scrub(value: Any, _depth: int = 0) -> Any:
    """Make `value` JSON-safe, secret-free and bounded."""
    if _depth > 6:
        return str(value)[:_MAX_STR]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        # NaN/Inf are not valid JSON.
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        out = _KEY_RE.sub(_REDACTED, value)
        live = os.environ.get("ANTHROPIC_API_KEY")
        if live and len(live) > 8 and live in out:
            out = out.replace(live, _REDACTED)
        return out if len(out) <= _MAX_STR else out[: _MAX_STR - 3] + "..."
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _scrub(asdict(value), _depth + 1)
    if isinstance(value, dict):
        return {str(k): _scrub(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_scrub(v, _depth + 1) for v in value]
    return _scrub(str(value), _depth + 1)


def _ordered(event: dict) -> dict:
    """Known fields first, in schema order; anything extra keeps its own order."""
    out = {k: event[k] for k in _FIELD_ORDER if k in event}
    out.update({k: v for k, v in event.items() if k not in out})
    return out


class Journal:
    """One journal per run. Thread-safe; the loop is single-threaded anyway."""

    def __init__(self, path: Path | str, run_id: str, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._fsync = fsync
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self.count = 0

    def emit(self, event: dict) -> dict:
        """Append one event. Returns the row as written (handy in tests)."""
        row = dict(event)
        row.setdefault("ts", utcnow())
        row.setdefault("run_id", self.run_id)
        row = _ordered({k: _scrub(v) for k, v in row.items() if v is not None})
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        assert "\n" not in line
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
            self.count += 1
        return row

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# Module-level default journal — this is the `emit(event)` of contracts.md §3.
# --------------------------------------------------------------------------

_default: Journal | None = None


def configure(run_dir: Path | str, run_id: str, *, fsync: bool = True) -> Journal:
    """Point the module-level `emit()` at `runs/<run_id>/journal.jsonl`."""
    global _default
    close()
    _default = Journal(Path(run_dir) / "journal.jsonl", run_id, fsync=fsync)
    return _default


def current() -> Journal | None:
    return _default


def emit(event: dict) -> None:
    """Append one JSON line to the configured journal. No-op if unconfigured."""
    if _default is not None:
        _default.emit(event)


def close() -> None:
    global _default
    if _default is not None:
        _default.close()
        _default = None


def read(path: Path | str) -> Iterator[dict]:
    """Stream a journal back. Skips a torn final line rather than raising —
    a `kill -9` mid-write must never make the run unreadable."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------
# Accounting — a reported deliverable, and it gates the Feasibility tier.
# --------------------------------------------------------------------------


class Accountant:
    """Cumulative tokens, wall-clock and iterations. Must be exact."""

    def __init__(self, *, token_budget: int | None = None, now: float | None = None) -> None:
        import time

        self._time = time.monotonic
        self.started_at = now if now is not None else self._time()
        self.wall_started_utc = utcnow()
        self.tokens_in = 0
        self.tokens_out = 0
        self.iterations = 0
        self.llm_calls = 0
        self.exec_seconds = 0.0
        self.token_budget = token_budget

    # -- updates -----------------------------------------------------------
    def add_tokens(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in += int(tokens_in or 0)
        self.tokens_out += int(tokens_out or 0)
        self.llm_calls += 1

    def add_exec(self, wall_s: float) -> None:
        self.exec_seconds += float(wall_s or 0.0)

    def tick_iteration(self) -> None:
        self.iterations += 1

    # -- reads -------------------------------------------------------------
    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    def elapsed_s(self) -> float:
        return self._time() - self.started_at

    def tokens_left(self, budget: int | None = None) -> int:
        budget = budget if budget is not None else self.token_budget
        if budget is None:
            return 1 << 30
        return max(0, budget - self.tokens_total)

    def snapshot(self) -> dict:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_total,
            "llm_calls": self.llm_calls,
            "iterations": self.iterations,
            "wall_s": round(self.elapsed_s(), 3),
            "exec_s": round(self.exec_seconds, 3),
        }

    def restore(self, snap: dict, *, elapsed_s: float = 0.0) -> None:
        """Rebuild after `--resume`. Wall-clock continues from prior elapsed."""
        self.tokens_in = int(snap.get("tokens_in", 0))
        self.tokens_out = int(snap.get("tokens_out", 0))
        self.llm_calls = int(snap.get("llm_calls", 0))
        self.iterations = int(snap.get("iterations", 0))
        self.exec_seconds = float(snap.get("exec_s", 0.0))
        self.started_at = self._time() - float(elapsed_s or snap.get("wall_s", 0.0))
