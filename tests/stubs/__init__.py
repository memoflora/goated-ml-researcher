"""Stub implementations of the frozen seams. Owner: A.

Everyone codes against these until the real module lands. They are also what
`smoke` mode runs on: 3 iterations, stubbed LLM, no dataset, under 60 seconds.

    from tests.stubs import StubAgent, StubExecutor, StubEvaluator, stub_task
"""

from tests.stubs.agent import StubAgent
from tests.stubs.evaluator import StubEvaluator
from tests.stubs.executor import StubExecutor
from tests.stubs.task import stub_task

__all__ = ["StubAgent", "StubExecutor", "StubEvaluator", "stub_task"]
