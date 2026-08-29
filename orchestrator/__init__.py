"""Autonomous ML research agent for KuaiRand-Pure (TechJam 2026, Track 2)."""
import sys

if sys.version_info < (3, 10):  # noqa: UP036 # pragma: no cover - a clear message beats a traceback
    raise SystemExit(
        f"This project needs Python 3.10 or newer (contracts.md pins 3.11); "
        f"you are on {sys.version.split()[0]} at {sys.executable}. "
        f"Create a venv with a newer interpreter:\n"
        f"    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    )

__all__ = ["contracts", "journal"]
