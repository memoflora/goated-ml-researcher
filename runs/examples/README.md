# Committed run artifacts

`runs/` is gitignored. That is right for working output and wrong for the parts the
organisers grade, so curated runs are copied here and committed.

**Three required deliverables live only in a run directory** (problem statement §2.5):

| # | Required | File in a run archived here |
|---|---|---|
| 3 | Run and iteration logs | `journal.jsonl` |
| 3b | Number of manual interventions | `interventions.md` |
| 4a | Final model output, starter-kit schema | `final/submission.csv` |

Until a run is archived here, the repository does not contain them, and
`docs/handover/05-deliverables.md` points at paths a reviewer cannot open.

## Adding a run

```bash
python tools/archive_run.py runs/<run_id>
git add runs/examples && git commit
```

The script is selective, and deliberately so. A full official run is a few hundred
megabytes — every node writes its own 170,588-row `submission.csv`, and so does every
final seed. Those are reproducible from the pipeline and are evidence of nothing. It
keeps the journal, config, state and summary; `interventions.md`; `RESULTS.md` and
`trajectory.png` when they exist; `final/submission.csv` and `best/submission.csv`; and
every node's `pipeline.py`. It prints what it copied and warns if the total is large.

It never writes back to the source run, and refuses to overwrite an existing archive
without `--force`.

## Which runs belong here

At minimum the ones the writeup cites. As of the last handover that is:

- **`r20260831-0724`** (gpt-4o, validation primary 0.58059) — the only run that produced
  both a defensible score and a valid 170,588-row submission. This one carries
  deliverable 4a.
- **`r20260831-0741`** (gpt-5.1, validation primary 0.59184) — the best clean score, and
  the source of the agent-hypothesis quotes still marked `TODO` in `docs/devpost.md`.
  It crashed during finalisation and has no submission.

Both were run on a Windows machine and exist nowhere in this repository. Whoever has
them needs to run the command above; nobody else can fill in those quotes, because they
can only come from `journal.jsonl` (`event: "proposal"`, field `hypothesis`).

Do not archive the leaked runs (`r20260831-0532`, `r20260831-0708`) as results. If they
are added at all, it is as evidence for the leak finding, and
`docs/handover/01-results.md` is the authority on which numbers may be quoted.

## Reading an archived run without opening the files

```bash
python -m orchestrator.report runs/examples/<run_id>   # writes RESULTS.md + trajectory.png
```

It reads `journal.jsonl` and nothing else, so it works on an archived run exactly as it
does on a live one — including one that crashed before writing a summary.
