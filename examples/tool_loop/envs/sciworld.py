"""ScienceWorld environment session over raw HTTP (AgentGym server, :36003).

Lifecycle: ``POST /create`` -> ``POST /reset {id, data_idx}`` -> ``POST /step
{id, action}``, with a real ``/close``. Unlike AlfWorld the server returns no
admissible-action list, so the action grammar is stated once in the instruction
instead of being appended to every observation.

Two response fields matter and they are NOT interchangeable: ``score`` is
cumulative task progress (0-100, verified live), while ``reward`` is the per-step
delta. The shared scorer judges the terminal state, so this session reports the
cumulative one, normalized to [0, 1].
"""

import os
from typing import Any

from examples.tool_loop.envs import http_client
from examples.tool_loop.envs.base import EnvError, StepResult

# reset() loads a game (slow); step() is fast. Same split as AlfWorld.
_RESET_TIMEOUT = float(os.environ.get("TOOL_LOOP_ENV_RESET_TIMEOUT", "300"))
_STEP_TIMEOUT = float(os.environ.get("TOOL_LOOP_ENV_STEP_TIMEOUT", "120"))

# The 23 action templates the environment accepts, taken from the action list the
# official AgentGym trajectories put in their own system message.
_ACTIONS = (
    "open/close OBJ, de/activate OBJ, connect OBJ to OBJ, disconnect OBJ, "
    "use OBJ [on OBJ], look around, look at OBJ, look in OBJ, read OBJ, "
    "move OBJ to OBJ, pick up OBJ, put down OBJ, pour OBJ into OBJ, "
    "dunk OBJ into OBJ, mix OBJ, go to LOC, eat OBJ, flush OBJ, focus on OBJ, "
    "wait, wait1, task, inventory"
)

_INSTRUCTION = (
    "You are an agent in ScienceWorld, a simulated science environment. Complete "
    "the task stated in the first observation. Available actions:\n"
    f"  {_ACTIONS}\n"
    'Note that "focus on OBJ" marks the object the task is about — several tasks '
    "award no credit until you focus on the correct thing. Emit exactly one "
    "action as:\n"
    '  <call name="step">{"action": "go to kitchen"}</call>'
)


def _normalize_score(raw: Any) -> float:
    """Map ScienceWorld's 0-100 score onto [0, 1]; clamp negatives to 0.

    ScienceWorld reports a *cumulative percentage* of task progress, and uses -100
    to mean "unrecoverable" (the simulator sets its completion flag when the score
    goes negative). Both ends are clamped so the shared scorer's ``reward >= 1.0``
    test means exactly "fully solved".
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return 0.0
    return min(value / 100.0, 1.0)


class SciWorldSession:
    """One ScienceWorld task, addressed by ``data_idx`` on the server."""

    def __init__(
        self,
        base_url: str,
        data_idx: int,
        *,
        session_id: int = 0,
        timeout: float | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._data_idx = int(data_idx)
        self._id = int(session_id)
        self._timeout = timeout
        self._created = False

    @property
    def instruction(self) -> str:
        return _INSTRUCTION

    def _t(self, default: float) -> float:
        return self._timeout if self._timeout is not None else default

    def reset(self) -> str:
        create = http_client.post(self._base, "/create", timeout=self._t(_RESET_TIMEOUT))
        if isinstance(create, dict) and "id" in create:
            self._id = int(create["id"])
        self._created = True
        data = http_client.post(
            self._base,
            "/reset",
            {"id": self._id, "data_idx": self._data_idx},
            timeout=self._t(_RESET_TIMEOUT),
        )
        # The goal lives in task_description, NOT in the observation, so the two
        # must be joined or the model never learns what it is being asked to do
        # (mirrors agentenv's own SciWorldEnvClient.reset).
        description = str(data.get("task_description") or "").strip()
        observation = str(data.get("observation") or "").strip()
        return f"{description}\n{observation}".strip()

    def step(self, action: str) -> StepResult:
        data = http_client.post(
            self._base, "/step", {"id": self._id, "action": action}, timeout=self._t(_STEP_TIMEOUT)
        )
        # "reward" is the per-step DELTA; "score" is cumulative progress. The
        # scorer judges the terminal state, so the cumulative value is the one to
        # report — a per-step delta of 0 on the winning move would read as failure.
        raw_score = data.get("score", 0.0)
        return StepResult(
            observation=str(data.get("observation") or ""),
            reward=_normalize_score(raw_score),
            done=bool(data.get("done", False)),
            info={"raw_score": raw_score, "step_reward": data.get("reward")},
        )

    def close(self) -> None:
        if not self._created:
            return
        try:
            http_client.post(self._base, "/close", {"id": self._id}, timeout=self._t(_STEP_TIMEOUT))
        except EnvError:
            pass  # best-effort teardown; never raise on close
        finally:
            self._created = False
