"""XML-style call protocol: parse the model's output, dispatch to a tool.

The model is asked to emit exactly one of:

- a tool call:  ``<call name="TOOL">{"arg": ...}</call>``  (args is a JSON object)
- a final answer: ``<final>ANSWER</final>``

``parse_action`` extracts whichever appears (final takes precedence); anything
else — no tag, malformed JSON — yields ``None`` so the loop can feed back a
format-error message. ``dispatch`` runs a parsed tool call against the registry
and returns the feedback text (all local, no API).
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from examples.tool_loop.tools import get_tool, list_tools

# <final>...</final> — non-greedy, dot matches newlines.
_FINAL_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL)
# <call name="tool">...</call> — capture the tool name and the raw inner body.
_CALL_RE = re.compile(r'<call\s+name\s*=\s*"([^"]+)"\s*>(.*?)</call>', re.DOTALL)


@dataclass
class Action:
    """A parsed tool call."""

    name: str
    args: dict[str, Any]


@dataclass
class Final:
    """A parsed final answer."""

    answer: str


def parse_action(model_text: str) -> Action | Final | None:
    """Parse the model output into a Final, an Action, or None (unparseable)."""
    final_match = _FINAL_RE.search(model_text)
    if final_match:
        return Final(answer=final_match.group(1).strip())

    call_match = _CALL_RE.search(model_text)
    if call_match:
        name = call_match.group(1).strip()
        body = call_match.group(2).strip()
        try:
            args = json.loads(body) if body else {}
        except (ValueError, TypeError):
            return None
        if not isinstance(args, dict):
            return None
        return Action(name=name, args=args)

    return None


def format_error_feedback() -> str:
    return (
        "Your output did not contain a valid action. Emit exactly one of:\n"
        '  <call name="TOOL">{"arg": value}</call>\n'
        "  <final>YOUR_ANSWER</final>\n"
        f"Available tools: {list_tools()}. The call body must be a JSON object."
    )


def dispatch(action: Action, example: Any) -> str:
    """Run a parsed tool call and return its feedback text."""
    tool = get_tool(action.name)
    if tool is None:
        return f"Unknown tool '{action.name}'. Available tools: {list_tools()}."
    try:
        return tool.fn(action.args, example)
    except Exception as e:  # a tool bug shouldn't crash the whole episode
        return f"Tool '{action.name}' raised {type(e).__name__}: {e}. Check your arguments."
