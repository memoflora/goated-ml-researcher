"""Fault: import outside requirements-pipeline.txt -> ErrorClass 'import'."""
import definitely_not_a_real_library_xyz  # noqa: F401
from _cli import args

args()
