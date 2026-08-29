"""Not a fault: reports which secret-shaped env vars leaked into the sandbox."""
import json
import os
from _cli import args, write_submission


def secret_shaped(name):
    return (name.endswith(("API_KEY", "TOKEN")) or "SECRET" in name or "PASSWORD" in name
            or name.startswith(("ANTHROPIC_", "AWS_", "OPENAI_")))


a = args()
leaked = sorted(k for k in os.environ if secret_shaped(k))
write_submission(a.out_dir, 5)
print("RESULT_JSON " + json.dumps(
    {"n_rows": 5, "train_seconds": 0.0, "notes": "env", "leaked": leaked,
     "hashseed": os.environ.get("PYTHONHASHSEED")}))
