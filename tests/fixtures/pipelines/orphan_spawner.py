"""Fault: spawns a long-lived grandchild then hangs. The grandchild must die with
the process group, or a six-hour run leaks a process per timeout."""
import subprocess
import sys
import time
from _cli import args

args()
child = subprocess.Popen([sys.executable, "-c",
                          "import time\nwhile True: time.sleep(0.05)"])
print("SPAWNED %d" % child.pid, flush=True)
while True:
    time.sleep(0.05)
