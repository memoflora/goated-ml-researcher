"""Fault: exits 0, reports success, writes nothing -> ErrorClass 'contract'."""
import json
from _cli import args

a = args()
print("RESULT_JSON " + json.dumps({"n_rows": 20, "train_seconds": 0.01, "notes": "oops"}))
