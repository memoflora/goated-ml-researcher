"""StubExecutor — canned ExecResults, including failures. Mirrors B's sandbox.

    run(node, *, split, seed, timeout_s, subsample=None) -> ExecResult

Driven by a script of outcomes, one consumed per call:

    "ok:0.6151"   scored run, stub validation primary 0.6151
    "err:syntax"  failed run with that ErrorClass

It writes a schema-valid `submission.csv` into the node workspace so the
validator and scorer downstream of it have a real file to look at, plus a
`stub_metrics.json` sidecar that StubEvaluator reads back.
"""

from __future__ import annotations

import json
from pathlib import Path

from orchestrator.contracts import ErrorClass, ExecResult, Node

#: A realistic trajectory: a weak first draft, a syntax failure, steady gains,
#: a timeout, then a plateau that trips the eps=0.002 / N=3 convergence rule.
DEFAULT_SCRIPT: tuple[str, ...] = (
    "ok:0.5904",
    "err:syntax",
    "ok:0.6042",
    "ok:0.6151",
    "err:timeout",
    "ok:0.6203",
    "ok:0.6288",
    "ok:0.6301",
    "ok:0.6309",
    "ok:0.6312",
    "ok:0.6315",
)

#: Never converges — for the 50-iteration endurance test.
CLIMBING_SCRIPT: tuple[str, ...] = tuple(f"ok:{0.58 + 0.004 * i:.4f}" for i in range(80))

_STDERR = {
    "syntax": 'File "pipeline.py", line 62\n    for row in rows\n                  ^\nSyntaxError: expected \':\'',
    "import": 'File "pipeline.py", line 8, in <module>\n    import torch_geometric\nModuleNotFoundError: No module named \'torch_geometric\'',
    "timeout": "pipeline.py exceeded the 1500s timeout; process group killed",
    "oom": "MemoryError: Unable to allocate 41.2 GiB for an array with shape (1141112, 4864)",
    "contract": "pipeline.py exited 0 but wrote no submission.csv",
    "eval": "submission.csv contains 12 NaN scores at rows 40..51",
    "runtime": 'File "pipeline.py", line 118, in fit\n    grad = x.T @ err\nValueError: matmul: shapes (1024,31) and (30,) not aligned',
    "data": "FileNotFoundError: data/kuairand-pure/log_standard_4_08_to_4_21_pure.csv",
}


class StubExecutor:
    def __init__(self, script: tuple[str, ...] | list[str] = DEFAULT_SCRIPT) -> None:
        self.script = list(script)
        self.calls = 0

    def _next(self) -> str:
        if self.calls < len(self.script):
            outcome = self.script[self.calls]
        else:  # exhausted: flat plateau at the last scored value
            last = [s for s in self.script if s.startswith("ok:")]
            outcome = last[-1] if last else "ok:0.6000"
        self.calls += 1
        return outcome

    def run(
        self,
        node: Node,
        *,
        split: str = "val",
        seed: int = 0,
        timeout_s: int = 1500,
        subsample: float | None = None,
    ) -> ExecResult:
        outcome = self._next()
        ws = Path(node.workspace)
        ws.mkdir(parents=True, exist_ok=True)

        if outcome.startswith("err:"):
            cls: ErrorClass = outcome.split(":", 1)[1]  # type: ignore[assignment]
            excerpt = _STDERR.get(cls, "unknown failure")
            (ws / "stderr.log").write_text(excerpt, encoding="utf-8")
            return ExecResult(
                ok=False,
                exit_code=124 if cls == "timeout" else 1,
                stdout_tail="",
                stderr_tail=excerpt,
                error_class=cls,
                error_excerpt=excerpt[:1500],
                result_json=None,
                artifacts={},
                wall_s=1500.0 if cls == "timeout" else 3.2,
                peak_rss_mb=41000.0 if cls == "oom" else 180.0,
            )

        value = float(outcome.split(":", 1)[1])
        n_rows = 124_909 if split == "val" else 170_588
        if subsample:
            n_rows = max(10, int(n_rows * subsample))
        sub = ws / "submission.csv"
        with sub.open("w", encoding="utf-8") as fh:
            fh.write("row_id,user_id,video_id,score\n")
            for row_id in range(min(n_rows, 200)):  # small on disk, honest schema
                fh.write(f"{row_id},{row_id % 97},{row_id % 313},{(row_id % 89) / 89:.6f}\n")
        # sidecar the StubEvaluator reads instead of really scoring
        metrics = {
            "gauc": round(value + 0.0658, 4),
            "ndcg@5": round(value - 0.0659, 4),
            "primary": value,
        }
        (ws / "stub_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        stdout = json.dumps({"n_rows": n_rows, "train_seconds": 41.7, "notes": "stub run"})
        (ws / "stdout.log").write_text(f"RESULT_JSON {stdout}\n", encoding="utf-8")
        return ExecResult(
            ok=True,
            exit_code=0,
            stdout_tail=f"RESULT_JSON {stdout}",
            stderr_tail="",
            error_class=None,
            error_excerpt=None,
            result_json=json.loads(stdout),
            artifacts={"submission": sub},
            wall_s=44.1,
            peak_rss_mb=920.0,
        )
