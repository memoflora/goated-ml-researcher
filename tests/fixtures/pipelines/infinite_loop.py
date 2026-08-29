"""Fault: never terminates -> ErrorClass 'timeout', process group killed."""
import time
from _cli import args

args()
print("training", flush=True)
while True:
    time.sleep(0.05)
