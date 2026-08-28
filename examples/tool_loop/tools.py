"""Tool registry for the multi-turn tool-feedback loop.

A "tool" is a pure-Python function ``(args: dict, example) -> str`` that, given
the arguments the model supplied in a ``<call>`` and the current example, returns
a feedback string appended back into the conversation. No API is involved — the
feedback is generated locally.

Adding a tool is one decorator, mirroring the ``register_engine`` pattern in
``src/gepa/oa/registry.py``:

    @register_tool("probe", "Probe a value and get distance-to-target feedback.")
    def _probe(args: dict, example) -> str:
        ...
        return "..."
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

FeedbackFn = Callable[[dict, Any], str]


@dataclass
class Tool:
    name: str
    fn: FeedbackFn
    description: str = ""


_TOOLS: dict[str, Tool] = {}


def register_tool(name: str, description: str = "") -> Callable[[FeedbackFn], FeedbackFn]:
    """Register a tool under ``name``. Returns the undecorated function unchanged."""

    def deco(fn: FeedbackFn) -> FeedbackFn:
        _TOOLS[name] = Tool(name=name, fn=fn, description=description)
        return fn

    return deco


def get_tool(name: str) -> Tool | None:
    return _TOOLS.get(name)


def list_tools() -> list[str]:
    return sorted(_TOOLS)


def tool_descriptions() -> str:
    """Human-readable catalog of registered tools (for prompts / docs)."""
    return "\n".join(f"- {t.name}: {t.description}" for t in (_TOOLS[n] for n in list_tools()))
