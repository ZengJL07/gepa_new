"""Score an env Episode by the environment's terminal reward + reflective feedback.

Scoring rules (AgentGym TextCraft / AlfWorld / ScienceWorld):
- ``format_errors >= _FORMAT_ERROR_LIMIT`` -> 0.0, even if the task was solved.
  The ``<call name="step">`` protocol is the contract; an agent that repeatedly
  breaks it is not credited for stumbling into the goal anyway.
- Task completed (``stop_reason=="done"`` and reward >= 1.0) -> 1.0.
- Everything else (done-but-no-reward, max_turns, token_budget, truncated, early
  ``<final>`` give-up) -> 0.0.

The feedback string summarizes the whole trajectory (budget *utilization*, the
full action sequence, last observation) so GEPA's reflection LM can improve the
initial prompt toward reaching the goal faster with fewer invalid actions and
strict XML.

It also states the dataset's reference solution for that task instance, on EVERY
outcome including successes — the same convention the math tasks use, where the
feedback always names the gold answer (aime_math/scoring.py). Only the expert
actions are included, not the expert's observations.
"""

from typing import Any

from examples.tool_loop.task_env import Episode

_MAX_OBS_CHARS = 400

# Episodes with this many malformed actions score 0 regardless of task outcome.
# Deliberately strict: the protocol is a single fixed tag and the model is told
# it verbatim, so 2+ violations means it never learned the contract. This gates
# the score only — the reflection LM additionally sees the exact count below and
# every offending output verbatim in SideInfo["trajectory"].
_FORMAT_ERROR_LIMIT = 2


def _last_observation(episode: Episode) -> str:
    for msg in reversed(episode.messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))[:_MAX_OBS_CHARS]
    return ""


def _action_sequence(episode: Episode) -> str:
    """The FULL action sequence — never truncated.

    A truncated list hid exactly what the reflection LM needs: with a 20-turn
    budget, most failures are inefficient exploration (opening every drawer in
    turn), and the tell is in the *middle* of the sequence. AlfWorld actions are
    short phrases ("go to drawer 4"), so a whole episode is a few hundred chars.
    """
    actions = [e["action"] for e in episode.trace if e.get("event") == "step" and "action" in e]
    return "; ".join(actions) if actions else "(none)"


def _budget_summary(episode: Episode) -> str:
    """Turn/token usage as utilization against the caps, not bare counts.

    The caps appear nowhere else the reflection LM can see — not in the prompt,
    not in the trajectory — so "turns=20" alone does not reveal that 20 was the
    wall. Reporting "20/20 (100%)" makes the ceiling actionable.
    """

    def _ratio(used: int, cap: int) -> str:
        return f"{used}/{cap}" + (f" ({used / cap:.0%})" if cap > 0 else "")

    return f"turns={_ratio(episode.turns_used, episode.max_turns)}, tokens={_ratio(episode.tokens_used, episode.max_total_tokens)}"


def _expert_reference(example: Any) -> str:
    """A reference solution for THIS task instance, for the reflection LM.

    Mirrors the math tasks, which always state the gold answer in their feedback
    (see aime_math/scoring.py) regardless of whether the attempt was right. Without
    it the reflection LM knows only that an episode was slow or stuck, never what a
    good route looks like, so it cannot do better than restate "be more efficient".

    Actions only — the dataset's observations are omitted, since a prompt can be
    taught to produce actions but pages of observation text would swamp the
    reflection prompt. Empty for test items, which ship without trajectories.
    """
    actions = getattr(example, "expert_actions", None) or ()
    if not actions:
        return ""
    return (
        f" Reference solution for this task ({len(actions)} actions): "
        + "; ".join(str(a) for a in actions)
        + "."
    )


def score_env_episode(episode: Episode, example: Any = None) -> tuple[float, str]:
    """Return (score, feedback). Higher is better; 1.0 iff the env task was solved."""
    summary = (
        f"stop_reason={episode.stop_reason}, {_budget_summary(episode)}, "
        f"tool_calls={episode.tool_calls}, "
        f"format_errors={episode.format_errors}, reward={episode.reward}"
    )
    actions = _action_sequence(episode)
    expert = _expert_reference(example)

    # Checked BEFORE the success case: a solved task with too many malformed
    # actions still scores 0, so this must not be shadowed by the return below.
    if episode.format_errors >= _FORMAT_ERROR_LIMIT:
        return 0.0, (
            f"Scored 0: {episode.format_errors} malformed actions (limit is {_FORMAT_ERROR_LIMIT}) — "
            f"this overrides the task outcome, which was stop_reason={episode.stop_reason} "
            f"reward={episode.reward}. {summary}. Actions: {actions}. "
            'Every turn must be exactly one <call name="step">{"action": "..."}</call> '
            "and nothing else."
            f"{expert}"
        )

    if episode.stop_reason == "done" and episode.reward >= 1.0:
        return 1.0, f"Task completed successfully. {summary}. Actions: {actions}.{expert}"

    if episode.stop_reason == "done":
        return 0.0, (
            f"The episode ended (done) but the task was not rewarded. {summary}. "
            f"Actions: {actions}. Last observation: {_last_observation(episode)}"
            f"{expert}"
        )

    if episode.stop_reason == "truncated":
        return 0.0, f"The model response was truncated (ran out of tokens). {summary}. Scored 0.{expert}"

    if episode.stop_reason == "final":
        return 0.0, (
            f"The model gave up early with <final> instead of solving the task. {summary}. "
            f"Actions: {actions}. Keep interacting via <call name=\"step\"> until done."
            f"{expert}"
        )

    if episode.stop_reason == "max_turns":
        return 0.0, (
            f"The episode hit the HARD TURN CAP of {episode.max_turns} turns without completing "
            f"the task. {summary}. Actions: {actions}. Last observation: {_last_observation(episode)} "
            f"The task must be finished within {episode.max_turns} actions, so guide the model to "
            "plan a route to the goal instead of exhaustively searching every container."
            f"{expert}"
        )

    if episode.stop_reason == "token_budget":
        return 0.0, (
            f"The episode hit the HARD TOKEN CAP of {episode.max_total_tokens} tokens without "
            f"completing the task, after only {episode.turns_used} turns. {summary}. "
            f"Actions: {actions}. Last observation: {_last_observation(episode)} "
            "This cap counts the model's own reasoning as well as the observations, so guide it "
            "to think briefly and act, rather than deliberating at length each turn."
            f"{expert}"
        )

    return 0.0, (
        f"Task not completed. {summary}. Actions: {actions}. "
        f"Last observation: {_last_observation(episode)}{expert}"
    )
