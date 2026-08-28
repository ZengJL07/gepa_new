"""Stateful environment sessions for the tool-feedback loop.

Unlike ``tools.py`` (pure functions ``fn(args, example) -> str``), these envs hold
a live session against an AgentGym HTTP server: ``reset`` opens a task, ``step``
sends an action and returns an observation + reward + done, ``close`` tears down.
"""

from examples.tool_loop.envs.base import EnvError, EnvSession, StepResult

__all__ = ["EnvError", "EnvSession", "StepResult"]
