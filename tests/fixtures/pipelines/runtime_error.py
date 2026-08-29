"""Fault: ordinary exception deep in a call stack -> ErrorClass 'runtime'."""
from _cli import args


def level_three(x):
    return x["missing_key"]


def level_two(x):
    return level_three(x)


def level_one(x):
    return level_two(x)


args()
print("training", flush=True)
level_one({"present": 1})
