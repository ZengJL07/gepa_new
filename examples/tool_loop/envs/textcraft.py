"""TextCraft environment session over raw HTTP (AgentGym server, default :36001).

Lifecycle: ``POST /create`` -> get a session ``id`` -> ``POST /reset {id, data_idx}``
opens the crafting task -> ``POST /step {id, action}`` runs one command. The server
matches ``craft``/``get``/``inventory`` directly, so the model's action string is
forwarded verbatim (no ``Action:`` prefix needed — DATASETS.md §4 / plan).

Server responses (env_wrapper.py):
  /create -> {id, observation, done, reward}
  /reset  -> {id, observation, done, reward}
  /step   -> {observation, reward, done}
"""

from examples.tool_loop.envs import http_client
from examples.tool_loop.envs.base import EnvError, StepResult

_INSTRUCTION = (
    "You are playing TextCraft. Craft the goal item by issuing commands. "
    "Valid commands:\n"
    "  get [<n>] <item>            -- obtain a base material\n"
    "  craft [<n>] <item> using <n> <item>, <n> <item>, ...  -- craft from a recipe\n"
    "  inventory                   -- list what you currently hold\n"
    "Emit exactly one command per turn as:\n"
    '  <call name="step">{"action": "get 1 iron ingot"}</call>'
)


class TextCraftSession:
    """One TextCraft task, addressed by ``data_idx`` on the server."""

    def __init__(self, base_url: str, data_idx: int, *, session_id: int = 0, timeout: float = 300.0) -> None:
        self._base = base_url.rstrip("/")
        self._data_idx = int(data_idx)
        self._id = int(session_id)
        self._timeout = timeout
        self._created = False

    @property
    def instruction(self) -> str:
        return _INSTRUCTION

    def reset(self) -> str:
        create = http_client.post(self._base, "/create", timeout=self._timeout)
        # The server assigns/echoes the session id; honor it if present.
        if isinstance(create, dict) and "id" in create:
            self._id = int(create["id"])
        self._created = True
        data = http_client.post(
            self._base, "/reset", {"id": self._id, "data_idx": self._data_idx}, timeout=self._timeout
        )
        return str(data.get("observation", ""))

    def step(self, action: str) -> StepResult:
        data = http_client.post(
            self._base, "/step", {"id": self._id, "action": action}, timeout=self._timeout
        )
        return StepResult(
            observation=str(data.get("observation", "")),
            reward=float(data.get("reward", 0.0) or 0.0),
            done=bool(data.get("done", False)),
            info={},
        )

    def close(self) -> None:
        if not self._created:
            return
        try:
            http_client.post(self._base, "/close", {"id": self._id}, timeout=self._timeout)
        except EnvError:
            pass  # best-effort teardown; never raise on close
        finally:
            self._created = False
