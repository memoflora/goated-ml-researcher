"""Fault: NaN / Inf in the score column -> ErrorClass 'eval'."""
import json
from _cli import args, write_submission

a = args()
write_submission(a.out_dir, 20, score=lambda i: "nan" if i == 3 else 0.5)
print("RESULT_JSON " + json.dumps({"n_rows": 20, "train_seconds": 0.01, "notes": "nan"}))
