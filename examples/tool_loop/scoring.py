"""Turn an Episode into a score in [0, 1] plus reflective feedback.

Scoring rules (guessing task):
- Correct final answer -> 1.0.
- Wrong/again unparseable final -> 0.0.
- Never finished (hit max_turns / token_budget / truncated) -> 0.0.

The feedback string summarizes the whole trajectory (turns used, which budget
was hit, format errors, tool-call count) so GEPA's reflection LM can improve the
initial prompt — e.g. steer it toward fewer turns, tighter token use, and strict
XML formatting.
"""

from typing import Any

from examples.tool_loop.task_env import Episode


def _target_of(example: Any) -> int | None:
    t = getattr(example, "target", None)
    return int(t) if t is not None else None


def score_episode(episode: Episode, example: Any) -> tuple[float, str]:
    """Return (score, feedback). Higher score is better."""
    target = _target_of(example)

    parts = [
        f"stop_reason={episode.stop_reason}",
        f"turns={episode.turns_used}",
        f"tokens={episode.tokens_used}",
        f"tool_calls={episode.tool_calls}",
        f"format_errors={episode.format_errors}",
    ]
    summary = ", ".join(parts)

    if episode.stop_reason == "truncated":
        return 0.0, f"The model response was truncated (ran out of tokens). {summary}. Reason unreliable; scored 0."

    if episode.stop_reason == "max_turns":
        return 0.0, (
            f"The model never submitted a <final> answer within {episode.turns_used} turns. {summary}. "
            "Guide it to converge faster and finish with <final>."
        )

    if episode.stop_reason == "token_budget":
        return 0.0, (
            f"The episode exhausted its token budget before a <final> answer. {summary}. "
            "Guide the model to be more concise and probe more efficiently (e.g. binary search)."
        )

    # stop_reason == "final"
    answer_raw = (episode.final_answer or "").strip()
    try:
        answer = int(answer_raw)
    except (ValueError, TypeError):
        return 0.0, (
            f"The model submitted a non-integer final answer '{answer_raw}'. {summary}. "
            "The final answer must be a single integer."
        )

    if target is not None and answer == target:
        return 1.0, f"Correct: found {target}. {summary}."

    return 0.0, (
        f"Incorrect final answer {answer}"
        + (f" (correct was {target})" if target is not None else "")
        + f". {summary}. Use the probe feedback more carefully before finalizing."
    )
