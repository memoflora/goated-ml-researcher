"""Copy a run's graded artifacts into `runs/examples/` so they can be committed.

    python tools/archive_run.py runs/r20260831-0741

`runs/*` is gitignored, which is right for working output and wrong for the parts the
organisers actually ask for. Problem statement §2.5 requires the run and iteration logs
(item 3), the manual-intervention count (3b) and the final submission in the starter-kit
schema (4a) — all of which live under `runs/<id>/` and none of which are in the repo
until they are copied here.

It is selective on purpose. A full official run is a few hundred megabytes: every node
writes its own 170,588-row `submission.csv`, and so does every final seed. Those are
reproducible from the pipeline and are not evidence of anything, so they are skipped.
What is kept is what a judge would want to read:

  * the journal, config, state and summary  — the run's own account of itself
  * interventions.md                        — deliverable 3b, and our autonomy claim
  * RESULTS.md / trajectory.png             — the results table, when they exist
  * final/submission.csv                    — deliverable 4a, the one CSV that matters
  * every node's pipeline.py                — what the agent actually wrote, per node

Nothing is ever written back to the source run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "runs" / "examples"

#: Run-level files copied verbatim when present. Absent ones are reported, not fatal:
#: a crashed run has no summary.json and is still worth archiving.
TOP_LEVEL = (
    "config.json",
    "journal.jsonl",
    "summary.json",
    "state.json",
    "interventions.md",
    "RESULTS.md",
    "trajectory.png",
)

#: Warn above this. GitHub starts complaining at 50 MB for a single file.
SIZE_WARN_MB = 25.0


def _display(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise — never raises.

    `Path.relative_to` throws for anything outside the repo, which a bare
    `--name`-less call never produces but a test or an absolute `dest_root` does.
    Losing an archive to a formatting error in the success message would be absurd.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def archive(run_dir: Path, dest_root: Path = EXAMPLES, *, name: str | None = None,
            force: bool = False) -> Path:
    if not run_dir.is_dir():
        raise SystemExit(f"not a run directory: {run_dir}")
    if not (run_dir / "journal.jsonl").is_file():
        raise SystemExit(
            f"{run_dir} has no journal.jsonl, so it is not a run directory. "
            "Point this at runs/<run_id>."
        )

    dest = dest_root / (name or run_dir.name)
    if dest.exists() and not force:
        raise SystemExit(f"{dest} already exists; pass --force to replace it")
    if dest.exists():
        shutil.rmtree(dest)

    copied: list[tuple[str, int]] = []
    missing: list[str] = []

    for fname in TOP_LEVEL:
        src = run_dir / fname
        if src.is_file():
            copied.append((fname, _copy(src, dest / fname)))
        else:
            missing.append(fname)

    # Deliverable 4a. `best/` is the validation winner; `final/` is what gets submitted.
    for sub in ("final/submission.csv", "best/submission.csv"):
        src = run_dir / sub
        if src.is_file():
            copied.append((sub, _copy(src, dest / sub)))
        else:
            missing.append(sub)

    # Every pipeline the agent wrote, and nothing else from the node workspaces.
    for src in sorted(run_dir.glob("nodes/*/pipeline.py")):
        copied.append((str(src.relative_to(run_dir)), _copy(src, dest / src.relative_to(run_dir))))
    for sub in ("best/pipeline.py", "final/pipeline.py"):
        src = run_dir / sub
        if src.is_file():
            copied.append((sub, _copy(src, dest / sub)))

    total = sum(n for _, n in copied)
    print(f"archived {_display(run_dir)} -> {_display(dest)}")
    for rel, n in copied:
        flag = "  <-- large" if n / 1e6 > SIZE_WARN_MB else ""
        print(f"  {n / 1e6:8.2f} MB  {rel}{flag}")
    print(f"  {total / 1e6:8.2f} MB  total, {len(copied)} files")
    if missing:
        print(f"  not present (fine unless you expected it): {', '.join(missing)}")
    if total / 1e6 > SIZE_WARN_MB:
        print(
            f"\nWARNING: {total / 1e6:.0f} MB is a lot to commit. Check that no per-node "
            "submission.csv slipped in before pushing."
        )
    print("\nNext:  git add runs/examples && git commit")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path, help="the run to archive, e.g. runs/r20260831-0741")
    ap.add_argument("--name", default=None, help="name under runs/examples/ (default: the run id)")
    ap.add_argument("--force", action="store_true", help="replace an existing archive")
    args = ap.parse_args(argv)
    archive(args.run_dir, name=args.name, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
