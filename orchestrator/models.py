"""`make models` — list the model ids the configured keys can actually reach.

OWNER: B. This exists so nobody has to guess a model id: a wrong one is a 404 four
hours into a scored run. Prints ids only, never a key.

Both configured providers are listed, primary and fallback, because a fallback whose
model id is wrong is a fallback that does not exist — and you find out only at the
moment the primary dies, which is the worst possible time to find out.
"""

from __future__ import annotations

import os
import sys

from .agent import (
    FALLBACK_LLM_ENV,
    FALLBACK_MODEL_ENV,
    AgentError,
    GeminiClient,
    OpenAIClient,
    load_dotenv,
    make_client,
    provider_name,
)


def list_models(client) -> list[str] | None:
    """Model ids for a client, or None if it has nothing to list."""
    if isinstance(client, OpenAIClient):
        return sorted(m.id for m in client._client.models.list())
    if isinstance(client, GeminiClient):
        return client.list_models()
    if type(client).__name__ == "Anthropic":
        return sorted(m.id for m in client.models.list(limit=100))
    return None


def report(label: str, client, model_env: str) -> int:
    """Print one provider's catalogue. Returns 1 if it could not be reached."""
    name = provider_name(client)
    try:
        ids = list_models(client)
    except Exception as exc:  # noqa: BLE001 - report it, do not crash the listing
        print(f"{label}: {name} — unreachable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if ids is None:
        print(f"{label}: {name} — no models to list")
        return 0

    configured = os.environ.get(model_env)
    print(f"{label}: {name}   ({len(ids)} models reachable)")
    for model_id in ids:
        marker = "  <- " + model_env if model_id == configured else ""
        print(f"  {model_id}{marker}")
    if not configured:
        print(f"  {model_env} is not set — add it to .env")
    elif configured not in ids:
        # Worth shouting about: it is exactly the 404 this command exists to prevent.
        print(f"  WARNING: {model_env}={configured!r} is not in this list", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    load_dotenv()
    try:
        primary = make_client()
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    status = report("primary ", primary, "TECHJAM_MODEL")

    choice = (os.environ.get(FALLBACK_LLM_ENV) or "").strip().lower()
    if not choice:
        print(f"\nno fallback configured ({FALLBACK_LLM_ENV} is unset)")
        return status
    try:
        fallback = make_client(provider=choice)
    except AgentError as exc:
        print(f"\nfallback: {choice} — unusable: {exc}", file=sys.stderr)
        return 1
    print()
    return status or report("fallback", fallback, FALLBACK_MODEL_ENV)


if __name__ == "__main__":
    raise SystemExit(main())
