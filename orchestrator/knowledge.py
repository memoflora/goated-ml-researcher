"""The idea bank: what the agent knows it could try, and what it must not.

Owner: D. `orchestrator/run.py` imports this module and calls `retrieve` on it, so the
public surface is module-level functions, not a class.

Retrieval is deliberately rule-based rather than semantic. With 32 curated ideas and one
call per iteration, embeddings would add a dependency, a failure mode and a token cost to
solve a problem we do not have. The rules are:

  1. drop ideas already tried
  2. drop ideas whose prerequisites are not yet met - do not propose PLE before MMoE
  3. when the budget is nearly spent, drop ideas too slow to finish
  4. serve the lowest tier that still has untried entries, in curated order
  5. include one idea from the next tier as a lookahead, so escalation arrives as a
     gradient rather than a cliff

A prerequisite counts as met if it was tried OR - for tier-0 prerequisites - if anything
has scored at all. Tier 0 is "have a working baseline", and a scored node proves that
whether or not the agent happened to cite the idea by id. Without this, the entire bank
sits behind `T0.reproduce-fm`: an agent that writes a perfect baseline but never names
that idea would never be offered tier 1 or above, and the tiering would be inert for the
whole run. Evidence beats bookkeeping.

Tier order encodes the organisers' own ranked list of untested directions, which inverts
the obvious instinct: ranking losses first, architecture swaps last. See
`references/starter-kit-findings.md`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.contracts import Idea

IDEAS_PATH = Path(__file__).parent / "ideas.yaml"

# With this many iterations or fewer left, stop proposing long shots: an idea that cannot
# finish is worse than no idea, because it burns a turn and leaves a broken pipeline.
SHORT_BUDGET_ITERS = 5
SHORT_BUDGET_MINUTES = 20


@dataclass(frozen=True)
class DeadEnd:
    """A claim measured false, with the number that refutes it.

    These are static across a run, so they ride in the cached system block rather than
    being re-sent every call. `prompts/system.md` states them; `test_knowledge.py` checks
    the two stay in sync.
    """

    id: str
    claim: str
    verdict: str


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "knowledge.py needs pyyaml; add it to requirements.txt"
        ) from exc
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=4)
def _bank(path: str | None = None) -> tuple[tuple[Idea, ...], tuple[DeadEnd, ...]]:
    blob = _load_yaml(Path(path) if path else IDEAS_PATH)
    ideas = tuple(
        Idea(
            id=str(raw["id"]),
            tier=int(raw["tier"]),
            title=str(raw["title"]),
            summary=" ".join(str(raw["summary"]).split()),
            citation=raw.get("citation") or None,
            est_minutes=int(raw.get("est_minutes", 15)),
            prerequisites=list(raw.get("prerequisites") or []),
        )
        for raw in blob.get("ideas", [])
    )
    dead = tuple(
        DeadEnd(
            id=str(raw["id"]),
            claim=" ".join(str(raw["claim"]).split()),
            verdict=" ".join(str(raw["verdict"]).split()),
        )
        for raw in blob.get("dead_ends", [])
    )
    return ideas, dead


def all_ideas(path: str | None = None) -> list[Idea]:
    return list(_bank(path)[0])


def dead_ends(path: str | None = None) -> list[DeadEnd]:
    return list(_bank(path)[1])


def retrieve(
    *,
    tried: list[str] | None = None,
    best_metrics: dict[str, float] | None = None,
    budget_left: int = 50,
    k: int = 5,
    path: str | None = None,
) -> list[Idea]:
    """Top-k ideas for this iteration.

    `budget_left` is iterations remaining, not tokens. `best_metrics` is used only as
    evidence that *something* has scored, which unlocks the tier-0 prerequisites; the
    values themselves are not read, since a score that moves inside noise is not a signal.

    Never raises. The loop treats an empty list as "propose your own", which is a fine
    outcome; a crash here would cost an iteration and possibly an intervention.
    """
    if k <= 0:
        return []
    try:
        ideas = _bank(path)[0]
    except Exception:  # noqa: BLE001 - no ideas is survivable, a crash is not
        return []

    done = set(tried or [])
    by_id = {i.id: i for i in ideas}
    # Anything scored means the "get a baseline running" tier is satisfied in fact, even
    # if the agent never cited it by id. Otherwise the whole bank deadlocks at tier 0.
    has_scored = bool(best_metrics)

    def met(prereq: str) -> bool:
        if prereq in done:
            return True
        target = by_id.get(prereq)
        return has_scored and target is not None and target.tier == 0

    pool = [i for i in ideas if i.id not in done and all(met(p) for p in i.prerequisites)]
    if budget_left <= SHORT_BUDGET_ITERS:
        affordable = [i for i in pool if i.est_minutes <= SHORT_BUDGET_MINUTES]
        # Only apply the gate if it leaves something; an empty list here would be worse.
        pool = affordable or pool
    if not pool:
        return []

    order = {idea.id: n for n, idea in enumerate(ideas)}  # curated order within a tier
    pool.sort(key=lambda i: (i.tier, order[i.id]))

    lowest = pool[0].tier
    current = [i for i in pool if i.tier == lowest]
    lookahead = [i for i in pool if i.tier > lowest]

    # Serve the working tier, then a taste of the next so the agent can see where the
    # bank is heading before the current tier is fully exhausted.
    picked = current[:k]
    if lookahead:
        if len(picked) < k:
            picked += lookahead[: k - len(picked)]
        elif k >= 3:
            # The working tier fills k on its own; spend one slot on the lookahead.
            picked = picked[: k - 1] + [lookahead[0]]
    return picked[:k]
