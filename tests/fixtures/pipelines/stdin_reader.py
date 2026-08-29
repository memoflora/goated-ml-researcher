"""Fault: prompts for input. Must not hang the run -> stdin is /dev/null."""
from _cli import args

args()
value = input("continue? ")
print("got", value)
