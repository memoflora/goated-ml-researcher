"""Fault: wrong CSV header -> ErrorClass 'contract'."""
import json
from _cli import args, write_submission

a = args()
write_submission(a.out_dir, 20, header="user_id,video_id,score")
print("RESULT_JSON " + json.dumps({"n_rows": 20, "train_seconds": 0.01, "notes": "hdr"}))
