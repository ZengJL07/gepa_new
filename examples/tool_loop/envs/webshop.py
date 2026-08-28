"""WebShop environment session over raw HTTP (AgentGym server, :36004).

Lifecycle: ``POST /create`` -> ``POST /reset {env_idx, session_id}`` -> ``POST
/step {env_idx, action}``. Note the field name is ``env_idx``, not ``id`` as in
the other three environments, and ``reset`` takes the task index as
``session_id``.

Two things differ from AlfWorld/ScienceWorld and shape this adapter:

1. Reward arrives ONLY on ``click[Buy Now]``; every other step returns 0.0. There
   is no partial progress signal mid-episode.
2. That terminal reward is a fraction in [0, 1] — the share of the goal's
   attributes, options and price constraint that the purchased product satisfies,
   times a product-type match factor. A full match is exactly 1.0.
"""

import os
from typing import Any

from examples.tool_loop.envs import http_client
from examples.tool_loop.envs.base import EnvError, StepResult

_RESET_TIMEOUT = float(os.environ.get("TOOL_LOOP_ENV_RESET_TIMEOUT", "300"))
_STEP_TIMEOUT = float(os.environ.get("TOOL_LOOP_ENV_STEP_TIMEOUT", "120"))

_INSTRUCTION = (
    "You are web shopping. Find and buy the product described in the first "
    "observation.\n"
    "Two action forms are accepted:\n"
    '  search[keywords]   only when a search bar is available\n'
    "  click[value]       only for a value listed in the observation's clickables\n"
    "Buy by clicking the product, then each required option, then Buy Now. Credit "
    "is proportional to how many of the requested attributes, options and the "
    "price limit the purchased item satisfies, so check options before buying.\n"
    "Emit exactly one action as:\n"
    '  <call name="step">{"action": "search[navy shorts]"}</call>'
)


def _with_actions(observation: str, available: dict[str, Any] | None) -> str:
    """Append the available actions, mirroring agentenv's own ``observe()``.

    WebShop's clickables change every page, and an action outside them is silently
    ignored by the server (it returns ``reward=0, done=False`` and an unchanged
    page), so the model cannot recover by guessing. Listing them is what makes the
    task tractable.
    """
    obs = str(observation or "")
    if not available:
        return obs
    clickables = available.get("clickables") or []
    parts = [obs, f"Available actions: has_search_bar={bool(available.get('has_search_bar'))}"]
    if clickables:
        parts.append("clickables: " + ", ".join(str(c) for c in clickables))
    return "\n".join(parts)


def _normalize_reward(raw: Any) -> float:
    """Clamp WebShop's fractional reward into [0, 1].

    Already a fraction of the goal satisfied, so no rescaling is needed — only
    defensive clamping, and 0.0 for anything unparseable.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return min(max(value, 0.0), 1.0)


class WebShopSession:
    """One WebShop task, addressed by ``session_id`` on the server."""

    def __init__(
        self,
        base_url: str,
        session_id: int,
        *,
        timeout: float | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        # The task index. Distinct from _env_idx, which is the server-side slot.
        self._session_id = int(session_id)
        self._env_idx = 0
        self._timeout = timeout
        self._created = False

    @property
    def instruction(self) -> str:
        return _INSTRUCTION

    def _t(self, default: float) -> float:
        return self._timeout if self._timeout is not None else default

    def _available_actions(self) -> dict[str, Any] | None:
        """Current page's actions. Best-effort: a failure here costs the action
        list for one turn, which is not worth aborting an episode over."""
        try:
            return http_client.get(
                self._base,
                "/available_actions",
                {"env_idx": self._env_idx},
                timeout=self._t(_STEP_TIMEOUT),
            )
        except EnvError:
            return None

    def reset(self) -> str:
        # /create returns a bare int (the server slot), not a dict.
        created = http_client.post(self._base, "/create", timeout=self._t(_RESET_TIMEOUT))
        if isinstance(created, dict):
            created = created.get("env_idx", created.get("id", 0))
        self._env_idx = int(created)
        self._created = True

        # /reset returns [observation, None]; session_id selects WHICH task.
        data = http_client.post(
            self._base,
            "/reset",
            {"env_idx": self._env_idx, "session_id": self._session_id},
            timeout=self._t(_RESET_TIMEOUT),
        )
        observation = data[0] if isinstance(data, list) else data
        return _with_actions(observation, self._available_actions())

    def step(self, action: str) -> StepResult:
        data = http_client.post(
            self._base,
            "/step",
            {"env_idx": self._env_idx, "action": action},
            timeout=self._t(_STEP_TIMEOUT),
        )
        reward = _normalize_reward(data.get("reward", 0.0))
        done = bool(data.get("done", False))
        available = None if done else self._available_actions()
        return StepResult(
            observation=_with_actions(data.get("state", ""), available),
            reward=reward,
            done=done,
            info={"raw_reward": data.get("reward"), "available_actions": available},
        )

    def close(self) -> None:
        # The AgentGym webshop server exposes no /close route; sessions are reaped
        # by its own 8000-slot ring buffer.
        return
