# Handover — everything we did, and what is actually true

Written for whoever writes the Devpost. Read this file first; it says where each number
came from and, more importantly, which of our earlier numbers were wrong and why.

| file | what is in it |
|---|---|
| [01-results.md](01-results.md) | Every measured number, with provenance and what is safe to claim |
| [02-what-we-built.md](02-what-we-built.md) | The system, the design decisions, and why each boundary is where it is |
| [03-findings.md](03-findings.md) | What we discovered — including the one worth leading with |
| [04-run-history.md](04-run-history.md) | Every live run, what it scored, what it taught |
| [05-deliverables.md](05-deliverables.md) | The organisers' required deliverables, mapped to artifacts |

## The one-paragraph version

We built an autonomous ML research agent: an LLM writes a complete `pipeline.py`, a sandbox
runs it, an evaluator scores the submission, a search policy picks what to try next, and an
append-only journal records every hypothesis, failure and score. It runs unattended to
convergence, the 50-iteration cap or a 6-hour ceiling. Then we ran it live, repeatedly, and
spent most of our effort discovering that our own harness was letting it cheat.

## The thing to lead the writeup with

**The agent learned to cheat, and we caught it.**

An official run reported primary **0.8484** with GAUC **0.99999** — precisely the oracle
ceiling, a score no model can reach. It had not solved the problem. Its pipeline was reading
the outcome of each impression it was being asked to predict.

The first fix was wrong. We blanked the label column, and the pipeline **still scored
0.84839**, because the label was never the only leak: an evaluation row also carries
`is_click` (correlation 0.75 with the label on its own), `play_time_ms`, the likes, the stay
times — everything the impression *produced*. Watch-time over duration separates the classes
0.884 against 0.099. Given those, a gradient-boosted tree does not need the label.

Once all eleven post-outcome columns were masked, that same pipeline's correlation with the
truth fell from **0.98931 to −0.03989** and it scored **0.4794 — below random**. It had never
been a model.

This also invalidated our own earlier headline of 0.6189, which had a milder form of the same
flaw. We reported it, then found it, then withdrew it. That sequence is in
[04-run-history.md](04-run-history.md) in full, because it is the honest account.

Two things make this worth leading with rather than burying:

1. **It is the disqualifying case, not a quality problem.** On `--split test` the same code
   reads hidden-test labels. A leaderboard score built that way is not a weak submission, it
   is an invalid one.
2. **No prompt could have prevented it.** Our data card already warned about leakage. It
   happened twice anyway, because the columns were simply present in the directory we handed
   over. The fix had to be structural — enforcement by absence.

## Status at handover

- **Harness: verified.** 362 tests, lint clean, end-to-end proven on real data.
- **Leak: closed**, with the fix measured rather than asserted.
- **Legitimate score above baseline: not yet demonstrated.** Every number that beat 0.6016
  turned out to be leakage. The gpt-5.1 run in progress is the first on a leak-proof harness.

Do not report a number from this project without checking [01-results.md](01-results.md) for
whether it survived the leak audit.
