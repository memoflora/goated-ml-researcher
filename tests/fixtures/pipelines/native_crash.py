"""Fault: a compiled extension dies with no Python traceback -> ErrorClass 'native_crash'.

Stands in for the LightGBM access violation that cost run r20260831-0741 its
submission. `ctypes.string_at(0)` dereferences a null pointer inside libc, so the
process dies on SIGSEGV exactly the way a crash inside a native learner does — and,
crucially, prints nothing Python can be asked to read.
"""
import ctypes

from _cli import args

args()
print("training", flush=True)   # proves the crash happened mid-run, not at import
ctypes.string_at(0)
