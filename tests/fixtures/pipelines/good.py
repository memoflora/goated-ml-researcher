"""Happy path: valid submission + a well-formed RESULT_JSON line."""
import json
from _cli import args, write_submission

a = args()
n = 20 if a.subsample is None else max(1, int(20 * a.subsample))
write_submission(a.out_dir, n, score=lambda i: round(1.0 / (i + 1), 6))
print(json.dumps({"split": a.split, "seed": a.seed}))
print("RESULT_JSON " + json.dumps({"n_rows": n, "train_seconds": 0.01, "notes": "fixture"}))
