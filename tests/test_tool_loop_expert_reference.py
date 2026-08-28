"""Feedback must state the dataset's reference solution, on every outcome.

Same convention as the math tasks, whose feedback always names the gold answer
(aime_math/scoring.py) whether or not the attempt was correct. Without it the
reflection LM sees only that an episode was slow or stuck and cannot do better
than restate "be more efficient".

Actions only: a prompt can be taught to produce actions, but the expert's
observations would swamp the reflection prompt.
"""

import pytest

from examples.tool_loop.agentgym_datasets import _clean_expert_action, _is_acknowledgement
from examples.tool_loop.agentgym_scoring import score_env_episode
from examples.tool_loop.task_env import Episode


class _Ex:
    def __init__(self, actions):
        self.expert_actions = list(actions)
        self.item_id = "sciworld_1"


def _episode(stop_reason, reward=0.0):
    return Episode(
        messages=[{"role": "user", "content": "obs"}],
        final_answer=None,
        turns_used=5,
        tokens_used=100,
        stop_reason=stop_reason,
        reward=reward,
        env_done=(stop_reason == "done"),
        max_turns=40,
        max_total_tokens=24000,
    )


ALL_OUTCOMES = ["done", "max_turns", "token_budget", "truncated", "final", "env_error"]


@pytest.mark.parametrize("stop_reason", ALL_OUTCOMES)
def test_reference_is_present_on_every_failure_outcome(stop_reason):
    _, feedback = score_env_episode(_episode(stop_reason), _Ex(["go north", "take key"]))
    assert "Reference solution" in feedback
    assert "go north; take key" in feedback


def test_reference_is_present_on_success_too():
    """Explicitly requested: include it even when the episode already scored 1.0."""
    score, feedback = score_env_episode(_episode("done", reward=1.0), _Ex(["go north"]))
    assert score == 1.0
    assert "Reference solution" in feedback


def test_reference_is_present_when_the_format_gate_fires():
    ep = _episode("done", reward=1.0)
    ep.format_errors = 2
    score, feedback = score_env_episode(ep, _Ex(["go north"]))
    assert score == 0.0
    assert "Reference solution" in feedback


def test_action_count_is_stated():
    """The reflection LM should be able to compare its length against the reference."""
    _, feedback = score_env_episode(_episode("max_turns"), _Ex(["a", "b", "c"]))
    assert "(3 actions)" in feedback


@pytest.mark.parametrize("example", [None, _Ex([])], ids=["no_example", "empty_actions"])
def test_no_reference_when_none_is_available(example):
    """Test items ship without trajectories; feedback must stay well-formed."""
    _, feedback = score_env_episode(_episode("max_turns"), example)
    assert "Reference solution" not in feedback
    assert feedback.strip()


def test_scoring_is_unchanged_by_the_reference():
    """The reference is feedback only — it must never move the score."""
    for stop in ALL_OUTCOMES:
        with_ref, _ = score_env_episode(_episode(stop), _Ex(["a"]))
        without, _ = score_env_episode(_episode(stop), None)
        assert with_ref == without


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain action", "plain action"),
        ("Thought: I should look.\nAction: look around", "look around"),
        ("Action: go to kitchen", "go to kitchen"),
        ("  Action:   move OBJ to OBJ  ", "move OBJ to OBJ"),
    ],
)
def test_action_extraction_strips_the_thought_wrapper(raw, expected):
    assert _clean_expert_action(raw) == expected


def test_boilerplate_acknowledgement_is_recognized():
    """Every item in all three datasets opens with this instead of an action."""
    assert _is_acknowledgement("OK. I'll follow your instructions and try my best")
    assert not _is_acknowledgement("go to kitchen")
