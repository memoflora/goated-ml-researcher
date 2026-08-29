"""Fault: allocates without bound -> ErrorClass 'oom'."""
import time
from _cli import args

args()
blocks = []
while True:
    blocks.append(bytearray(32 * 1024 * 1024))
    for b in blocks[-1:]:
        b[::4096] = b"x" * len(b[::4096])   # touch pages so RSS actually grows
    time.sleep(0.02)
