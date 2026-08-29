"""`make models` — list the model ids the configured key can actually reach.

OWNER: B. This exists so nobody has to guess a model id: a wrong one is a 404 four
hours into a scored run. Prints ids only, never the key.
"""

from __future__ import annotations

import os
import sys

from .agent import AgentError, OpenAIClient, load_dotenv, make_client


def main() -> int:
    load_dotenv()
    try:
        client = make_client()
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if isinstance(client, OpenAIClient):
        ids = sorted(m.id for m in client._client.models.list())
        provider = "openai"
    elif type(client).__name__ == "Anthropic":
        ids = sorted(m.id for m in client.models.list(limit=100))
        provider = "anthropic"
    else:
        print("stub client: no models to list")
        return 0

    print(f"provider: {provider}   ({len(ids)} models reachable)")
    for model_id in ids:
        print(f"  {model_id}")
    current = os.environ.get("TECHJAM_MODEL")
    print(f"\nTECHJAM_MODEL is currently {current!r}" if current
          else "\nTECHJAM_MODEL is not set — add it to .env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
