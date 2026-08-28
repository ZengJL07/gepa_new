"""Single-example multi-turn rollout with a dual budget (turns + total tokens).

``run_episode`` drives the loop: send messages to the model, parse its output,
either finish, feed back a tool result, or feed back a format error — and repeat
until the model finishes or a budget is hit. Both the model's output AND the
tool-feedback text count toward ``max_total_tokens`` (the token limit "includes
tool calls", per requirements).

The model is injected as ``generate(messages) -> str`` so tests can drive the
loop with a scripted stub and no network. ``count_tokens`` is likewise injected
(defaults to a litellm-backed counter) to keep the loop pure and testable.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from examples.tool_loop.envs.base import EnvSession
from examples.tool_loop.protocol import Action, Final, dispatch, format_error_feedback, parse_action

GenerateFn = Callable[[list[dict[str, str]]], str]
CountFn = Callable[[str], int]


@dataclass
class Episode:
    messages: list[dict[str, str]]
    final_answer: str | None
    turns_used: int
    tokens_used: int
    stop_reason: str  # "final" | "max_turns" | "token_budget" | "truncated" | "done"
    format_errors: int = 0
    tool_calls: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0  # cumulative/terminal env reward (env episodes only)
    env_done: bool = False  # whether the env signaled task completion
    # The budget this episode ran under. Carried on the Episode so scorers can
    # report *utilization* ("18/20 turns") rather than a bare count — the caps
    # appear nowhere else in the reflective feedback, so without them the
    # reflection LM sees "turns=20" and cannot tell that 20 was the wall.
    max_turns: int = 0
    max_total_tokens: int = 0


def _default_counter(model: str) -> CountFn:
    """A litellm-backed token counter bound to ``model`` (lazy import)."""
    import litellm

    def _count(text: str) -> int:
        try:
            return int(litellm.token_counter(model=model, text=text or ""))
        except Exception:
            # Fallback: rough word-based estimate if litellm can't map the model.
            return len((text or "").split())

    return _count


def run_episode(
    generate: GenerateFn,
    initial_prompt: str,
    example: Any,
    max_turns: int,
    max_total_tokens: int,
    *,
    count_tokens: CountFn | None = None,
    truncated: Callable[[], bool] | None = None,
) -> Episode:
    """Run the tool-feedback loop for one example.

    Args:
        generate: model callable, ``messages -> assistant_text``.
        initial_prompt: the candidate being optimized; used as the system message.
        example: task instance, passed to tools for local feedback.
        max_turns: hard cap on model calls.
        max_total_tokens: cap on cumulative tokens (model output + tool feedback).
        count_tokens: token counter; defaults to a litellm counter on "gpt-4o".
        truncated: optional predicate — if it returns True after a model call, the
            episode ends with stop_reason="truncated" (wired to TruncationTrackingLM).
    """
    count = count_tokens or _default_counter("gpt-4o")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": initial_prompt},
        {"role": "user", "content": getattr(example, "input", str(example))},
    ]
    trace: list[dict[str, Any]] = []
    tokens_used = 0
    format_errors = 0
    tool_calls = 0

    turn = 0

    def _done(stop_reason: str, *, final_answer: str | None = None) -> Episode:
        """Snapshot the loop state as an Episode (reads locals at call time)."""
        return Episode(
            messages,
            final_answer,
            turn,
            tokens_used,
            stop_reason,
            format_errors,
            tool_calls,
            trace,
            max_turns=max_turns,
            max_total_tokens=max_total_tokens,
        )

    while True:
        if turn >= max_turns:
            return _done("max_turns")
        if tokens_used >= max_total_tokens:
            return _done("token_budget")

        output = generate(messages)
        turn += 1
        tokens_used += count(output)
        messages.append({"role": "assistant", "content": output})

        if truncated is not None and truncated():
            trace.append({"turn": turn, "event": "truncated", "output": output})
            return _done("truncated")

        action = parse_action(output)

        if isinstance(action, Final):
            trace.append({"turn": turn, "event": "final", "answer": action.answer})
            return _done("final", final_answer=action.answer)

        if isinstance(action, Action):
            tool_calls += 1
            feedback = dispatch(action, example)
            trace.append({"turn": turn, "event": "call", "name": action.name, "args": action.args})
        else:
            format_errors += 1
            feedback = format_error_feedback()
            trace.append({"turn": turn, "event": "format_error", "output": output})

        # Tool/format feedback counts toward the token budget too.
        tokens_used += count(feedback)
        messages.append({"role": "user", "content": feedback})


def _env_format_error() -> str:
    return (
        "Your output did not contain a valid action. Emit exactly one action as:\n"
        '  <call name="step">{"action": "YOUR ACTION"}</call>\n'
        "The call body must be a JSON object with a string \"action\" field."
    )


def run_env_episode(
    generate: GenerateFn,
    initial_prompt: str,
    session: EnvSession,
    *,
    max_turns: int,
    max_total_tokens: int,
    count_tokens: CountFn | None = None,
    truncated: Callable[[], bool] | None = None,
) -> Episode:
    """Run the tool-feedback loop against a stateful env ``session``.

    Mirrors :func:`run_episode` (dual budget, token counting incl. feedback, trace,
    truncation) but each turn dispatches ``<call name="step">{"action": ...}</call>``
    to ``session.step`` and terminates on the env's ``done`` flag. Score comes from
    the env reward, not a ``<final>`` answer. ``session.close()`` is always called.
    """
    count = count_tokens or _default_counter("gpt-4o")

    reward = 0.0
    tokens_used = 0
    format_errors = 0
    tool_calls = 0
    turn = 0
    try:
        obs = session.reset()
        system_content = f"{initial_prompt}\n\n{session.instruction}".strip()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": obs},
        ]
        trace: list[dict[str, Any]] = []
        tokens_used += count(obs)

        def _done(stop_reason: str, *, final_answer: str | None = None, env_done: bool = False) -> Episode:
            """Snapshot the loop state as an Episode (reads locals at call time)."""
            return Episode(
                messages,
                final_answer,
                turn,
                tokens_used,
                stop_reason,
                format_errors,
                tool_calls,
                trace,
                reward,
                env_done,
                max_turns,
                max_total_tokens,
            )

        while True:
            if turn >= max_turns:
                return _done("max_turns")
            if tokens_used >= max_total_tokens:
                return _done("token_budget")

            output = generate(messages)
            turn += 1
            tokens_used += count(output)
            messages.append({"role": "assistant", "content": output})

            if truncated is not None and truncated():
                trace.append({"turn": turn, "event": "truncated", "output": output})
                return _done("truncated")

            action = parse_action(output)

            if isinstance(action, Final):
                # Optional early give-up / self-report; not required to finish.
                trace.append({"turn": turn, "event": "final", "answer": action.answer})
                return _done("final", final_answer=action.answer)

            if isinstance(action, Action) and action.name == "step" and isinstance(action.args.get("action"), str):
                tool_calls += 1
                result = session.step(action.args["action"])
                reward = max(reward, result.reward)
                feedback = result.observation
                trace.append({"turn": turn, "event": "step", "action": action.args["action"], "reward": result.reward, "done": result.done})
                if result.done:
                    tokens_used += count(feedback)
                    messages.append({"role": "user", "content": feedback})
                    # Stop client-side on done (AlfWorld still accepts step after done).
                    return _done("done", env_done=True)
            else:
                format_errors += 1
                feedback = _env_format_error()
                trace.append({"turn": turn, "event": "format_error", "output": output})

            tokens_used += count(feedback)
            messages.append({"role": "user", "content": feedback})
    finally:
        session.close()
