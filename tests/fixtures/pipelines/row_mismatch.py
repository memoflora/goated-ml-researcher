"""Fault: CSV row count contradicts RESULT_JSON.n_rows -> ErrorClass 'contract'."""
import json
from _cli import args, write_submission

a = args()
write_submission(a.out_dir, 11)
print("RESULT_JSON " + json.dumps({"n_rows": 20, "train_seconds": 0.01, "notes": "short"}))
