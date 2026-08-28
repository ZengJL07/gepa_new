"""Environment-session contract for the tool loop.

A ``EnvSession`` is a stateful handle on one task instance: ``reset()`` returns the
opening observation (used as the first user message), ``step(action)`` sends a raw
action string and returns a ``StepResult``, and ``close()`` releases the session.

This is deliberately minimal so the loop in ``task_env.run_env_episode`` only needs
``reset``/``step``/``done``/``reward``. Concrete envs (TextCraft, AlfWorld) implement
it over raw HTTP; tests implement it in memory (``FakeEnv``).
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class EnvError(RuntimeError):
    """Raised when the environment server returns an error payload or bad status."""


@dataclass
class StepResult:
    """Outcome of one ``step``: feedback text, reward, terminal flag, and extras."""

    observation: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EnvSession(Protocol):
    """A stateful task session. Implementations must be safe to ``close`` twice."""

    @property
    def instruction(self) -> str:
        """Fixed environment description (action syntax) to seed the prompt."""
        ...

    def reset(self) -> str:
        """Open the task and return the initial observation (first user message)."""
        ...

    def step(self, action: str) -> StepResult:
        """Send a raw action string; return observation/reward/done."""
        ...

    def close(self) -> None:
        """Release the session. Must be idempotent and never raise."""
        ...
