"""AlfWorld environment session over raw HTTP (AgentGym server, default :36002).

Lifecycle: ``POST /create`` -> get session ``id`` -> ``POST /reset {id, game, world_type}``
opens the household task -> ``POST /step {id, action}``. Each observation is suffixed
with the admissible actions (mirrors agentenv's ``observe()``), and the model must pick
one of them.

Server responses (env_wrapper.py):
  /create -> {id}
  /reset  -> {id, observation, available_actions, done, reward}
  /step   -> {observation, reward, available_actions, done}

The server has NO /close route (DATASETS.md §8), so ``close`` is a no-op. It also
still accepts ``step`` after ``done``; the loop stops on ``done`` client-side.
"""

import os

from examples.tool_loop.envs import http_client
from examples.tool_loop.envs.base import StepResult

# Request timeouts, split because the two calls have very different profiles.
# /reset can legitimately take tens of seconds (the first visit to a game compiles
# a TextWorld game), while /step is fast — but the server serializes /step in one
# uvicorn process, so under 15 concurrent episodes a step waits behind others.
# The old blanket 300s meant a single wedged call parked a worker for five
# minutes, which is what "seems stuck, then errors" looked like.
_RESET_TIMEOUT = float(os.environ.get("TOOL_LOOP_ENV_RESET_TIMEOUT", "300"))
_STEP_TIMEOUT = float(os.environ.get("TOOL_LOOP_ENV_STEP_TIMEOUT", "120"))

_INSTRUCTION = (
    "You are an agent in a simulated household (AlfWorld). Complete the task described "
    "in the first observation. Each turn you are shown AVAILABLE ACTIONS; choose exactly "
    "one of them and emit it as:\n"
    '  <call name="step">{"action": "go to desk 1"}</call>'
)


def _with_actions(observation: str, available_actions: list[str]) -> str:
    """Append the admissible-action list to the observation (agentenv observe() format)."""
    obs = str(observation or "")
    if available_actions:
        obs = f"{obs}\nAVAILABLE ACTIONS: {','.join(str(a) for a in available_actions)}"
    return obs


class AlfWorldSession:
    """One AlfWorld task, addressed by ``game`` index on the server."""

    def __init__(
        self,
        base_url: str,
        game: int,
        *,
        session_id: int = 0,
        world_type: str = "Text",
        timeout: float | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._game = int(game)
        self._id = int(session_id)
        self._world_type = world_type
        # None => per-call defaults (reset is slow, step is not). An explicit
        # value overrides both, which tests rely on.
        self._timeout = timeout

    @property
    def instruction(self) -> str:
        return _INSTRUCTION

    def _t(self, default: float) -> float:
        return self._timeout if self._timeout is not None else default

    def reset(self) -> str:
        create = http_client.post(self._base, "/create", timeout=self._t(_RESET_TIMEOUT))
        if isinstance(create, dict) and "id" in create:
            self._id = int(create["id"])
        data = http_client.post(
            self._base,
            "/reset",
            {"id": self._id, "game": self._game, "world_type": self._world_type},
            timeout=self._t(_RESET_TIMEOUT),
        )
        return _with_actions(data.get("observation", ""), data.get("available_actions", []))

    def step(self, action: str) -> StepResult:
        data = http_client.post(
            self._base, "/step", {"id": self._id, "action": action}, timeout=self._t(_STEP_TIMEOUT)
        )
        available = data.get("available_actions", [])
        return StepResult(
            observation=_with_actions(data.get("observation", ""), available),
            reward=float(data.get("reward", 0.0) or 0.0),
            done=bool(data.get("done", False)),
            info={"available_actions": available},
        )

    def close(self) -> None:
        # AlfWorld server exposes no /close route (DATASETS.md §8); nothing to do.
        return
