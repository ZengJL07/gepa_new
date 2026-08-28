"""Offline tests for the multi-turn tool-feedback loop example (no network).

Covers the four self-contained pieces:
- tools: the register_tool decorator + get_tool/list_tools + dispatch on unknown.
- protocol: parse_action on valid/invalid <call>/<final>.
- task_env.run_episode: the dual budget (turns + tokens-incl-feedback), format
  retries, final termination, and truncation.
- scoring.score_episode: score per stop_reason and correctness.

The model is a scripted stub (`ScriptedModel`) so the loop runs deterministically
with no API calls.
"""

from examples.tool_loop.datasets import GuessExample, load_splits, make_dataset
from examples.tool_loop.protocol import Action, Final, dispatch, parse_action
from examples.tool_loop.scoring import score_episode
from examples.tool_loop.task_env import run_episode
from examples.tool_loop.tools import get_tool, list_tools, register_tool


class ScriptedModel:
    """Yields pre-scripted outputs turn by turn (ignores messages)."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def __call__(self, messages):
        out = self._outputs[self.calls] if self.calls < len(self._outputs) else "<noop/>"
        self.calls += 1
        return out


def _example(target=42, lo=1, hi=100):
    return GuessExample(input="find it", lo=lo, hi=hi, target=target)


# --- tools registry -------------------------------------------------------


def test_register_tool_and_lookup():
    @register_tool("_unit_echo", "echoes back")
    def _echo(args, example):
        return f"echo:{args.get('x')}"

    assert "_unit_echo" in list_tools()
    tool = get_tool("_unit_echo")
    assert tool is not None
    assert tool.description == "echoes back"
    assert tool.fn({"x": 7}, None) == "echo:7"


def test_get_tool_unknown_returns_none():
    assert get_tool("does_not_exist") is None


def test_dispatch_unknown_tool_reports_available():
    feedback = dispatch(Action(name="nope", args={}), _example())
    assert "nope" in feedback
    assert "probe" in feedback  # available tools listed


def test_probe_tool_registered_by_datasets_import():
    assert "probe" in list_tools()


# --- protocol -------------------------------------------------------------


def test_parse_final_takes_precedence():
    action = parse_action('<call name="probe">{"value": 5}</call><final>42</final>')
    assert isinstance(action, Final)
    assert action.answer == "42"


def test_parse_valid_call():
    action = parse_action('<call name="probe">{"value": 50}</call>')
    assert isinstance(action, Action)
    assert action.name == "probe"
    assert action.args == {"value": 50}


def test_parse_call_empty_body_is_empty_dict():
    action = parse_action('<call name="probe"></call>')
    assert isinstance(action, Action)
    assert action.args == {}


def test_parse_invalid_json_returns_none():
    assert parse_action('<call name="probe">{not json}</call>') is None


def test_parse_non_object_json_returns_none():
    assert parse_action('<call name="probe">[1, 2, 3]</call>') is None


def test_parse_no_tag_returns_none():
    assert parse_action("I am just thinking out loud.") is None


# --- run_episode ----------------------------------------------------------


def _fixed_counter(n):
    """Every text counts as ``n`` tokens (deterministic budget accounting)."""
    return lambda text: n


def test_episode_finishes_on_final():
    model = ScriptedModel(['<call name="probe">{"value": 42}</call>', "<final>42</final>"])
    ep = run_episode(model, "sys", _example(42), max_turns=6, max_total_tokens=10_000, count_tokens=_fixed_counter(1))
    assert ep.stop_reason == "final"
    assert ep.final_answer == "42"
    assert ep.tool_calls == 1
    assert ep.turns_used == 2


def test_episode_format_error_then_retry():
    model = ScriptedModel(["garbage with no tags", "<final>7</final>"])
    ep = run_episode(model, "sys", _example(7), max_turns=6, max_total_tokens=10_000, count_tokens=_fixed_counter(1))
    assert ep.stop_reason == "final"
    assert ep.format_errors == 1
    # A format-error feedback message must have been appended before the retry.
    assert any(m["role"] == "user" and "valid action" in m["content"] for m in ep.messages)


def test_episode_hits_max_turns():
    model = ScriptedModel(['<call name="probe">{"value": 1}</call>'] * 10)
    ep = run_episode(model, "sys", _example(42), max_turns=3, max_total_tokens=10_000, count_tokens=_fixed_counter(1))
    assert ep.stop_reason == "max_turns"
    assert ep.final_answer is None
    assert ep.turns_used == 3


def test_episode_hits_token_budget_and_feedback_counts():
    # Each model output = 10 tokens, each feedback = 10 tokens. Budget 25 -> after
    # one round (10 output + 10 feedback = 20) the next turn's pre-check trips at 20 >= ... no;
    # set budget so ONLY the feedback pushes it over, proving feedback is counted.
    model = ScriptedModel(['<call name="probe">{"value": 1}</call>'] * 10)
    ep = run_episode(model, "sys", _example(42), max_turns=99, max_total_tokens=15, count_tokens=_fixed_counter(10))
    assert ep.stop_reason == "token_budget"
    # One turn ran: 10 (output) + 10 (feedback) = 20 tokens, exceeding 15 only
    # because the tool feedback was included in the accounting.
    assert ep.tokens_used == 20
    assert ep.turns_used == 1


def test_episode_token_budget_output_alone_under_limit():
    # If feedback were NOT counted, 10 tokens/turn would allow many turns under a
    # budget of 15. Because feedback IS counted, we stop after one round.
    model = ScriptedModel(['<call name="probe">{"value": 1}</call>'] * 10)
    ep = run_episode(model, "sys", _example(42), max_turns=99, max_total_tokens=15, count_tokens=_fixed_counter(10))
    assert ep.turns_used == 1  # not 2+, proving feedback tokens gated the loop


def test_episode_truncation_ends_episode():
    flag = {"hit": False}

    def model(messages):
        flag["hit"] = True
        return "<final>42</final>"

    ep = run_episode(
        model,
        "sys",
        _example(42),
        max_turns=6,
        max_total_tokens=10_000,
        count_tokens=_fixed_counter(1),
        truncated=lambda: flag["hit"],
    )
    assert ep.stop_reason == "truncated"
    assert ep.final_answer is None


# --- scoring --------------------------------------------------------------


def test_score_correct_final():
    model = ScriptedModel(["<final>42</final>"])
    ep = run_episode(model, "sys", _example(42), max_turns=6, max_total_tokens=10_000, count_tokens=_fixed_counter(1))
    score, feedback = score_episode(ep, _example(42))
    assert score == 1.0
    assert "Correct" in feedback


def test_score_wrong_final():
    model = ScriptedModel(["<final>7</final>"])
    ep = run_episode(model, "sys", _example(42), max_turns=6, max_total_tokens=10_000, count_tokens=_fixed_counter(1))
    score, feedback = score_episode(ep, _example(42))
    assert score == 0.0
    assert "Incorrect" in feedback


def test_score_non_integer_final():
    model = ScriptedModel(["<final>forty-two</final>"])
    ep = run_episode(model, "sys", _example(42), max_turns=6, max_total_tokens=10_000, count_tokens=_fixed_counter(1))
    score, feedback = score_episode(ep, _example(42))
    assert score == 0.0
    assert "non-integer" in feedback


def test_score_max_turns_is_zero():
    model = ScriptedModel(['<call name="probe">{"value": 1}</call>'] * 10)
    ep = run_episode(model, "sys", _example(42), max_turns=2, max_total_tokens=10_000, count_tokens=_fixed_counter(1))
    score, feedback = score_episode(ep, _example(42))
    assert score == 0.0
    assert "never submitted" in feedback


def test_score_token_budget_is_zero():
    model = ScriptedModel(['<call name="probe">{"value": 1}</call>'] * 10)
    ep = run_episode(model, "sys", _example(42), max_turns=99, max_total_tokens=5, count_tokens=_fixed_counter(10))
    score, feedback = score_episode(ep, _example(42))
    assert score == 0.0
    assert "token budget" in feedback


# --- datasets -------------------------------------------------------------


def test_make_dataset_is_reproducible():
    a = make_dataset(5, seed=123)
    b = make_dataset(5, seed=123)
    assert [x.target for x in a] == [x.target for x in b]
    assert all(x.lo <= x.target <= x.hi for x in a)


def test_load_splits_are_disjoint_seeds():
    train, val, test = load_splits(train_n=4, val_n=3, test_n=2, seed=0)
    assert (len(train), len(val), len(test)) == (4, 3, 2)
    # Different seeds per split -> target sequences should not be identical.
    assert [x.target for x in train[:2]] != [x.target for x in val[:2]]
