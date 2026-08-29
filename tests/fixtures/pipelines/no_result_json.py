"""Fault: valid CSV but no RESULT_JSON line -> ErrorClass 'contract'."""
from _cli import args, write_submission

a = args()
write_submission(a.out_dir, 20)
print("done training")
