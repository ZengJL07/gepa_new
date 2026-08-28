"""Offline tests for the AgentGym (TextCraft / AlfWorld) tool-loop adaptation.

No network / no servers: a scripted model drives the loop and a in-memory
``FakeEnv`` implements the EnvSession contract. Covers:
- run_env_episode: done termination, max_turns, token budget incl. env observation,
  format-error retry, done stops immediately (no extra step), close() always called.
- score_env_episode: solved -> 1.0; each failure bucket -> 0.0 with feedback.
- agentgym_datasets: id -> env-index mapping (alfworld train lookup / test tail,
  textcraft remap / test tail) and seeded train/val split.
"""

import json

import pytest

from examples.tool_loop.agentgym_datasets import _resolve_index, load_agentgym_splits
from examples.tool_loop.agentgym_scoring import _FORMAT_ERROR_LIMIT, score_env_episode
from examples.tool_loop.envs import http_client
from examples.tool_loop.envs.base import EnvError, StepResult
from examples.tool_loop.envs.textcraft import TextCraftSession
from examples.tool_loop.task_env import Episode, run_env_episode


class ScriptedModel:
    """Yields pre-scripted outputs turn by turn (ignores messages)."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def __call__(self, messages):
        out = self._outputs[self.calls] if self.calls < len(self._outputs) else "<noop/>"
        self.calls += 1
        return out


class FakeEnv:
    """In-memory EnvSession: reset returns a fixed obs; step[k] scripted; done at done_at."""

    def __init__(self, *, done_at=None, reward_on_done=1.0, obs="OBS", step_obs="STEP_OBS"):
        self._done_at = done_at
        self._reward_on_done = reward_on_done
        self._obs = obs
        self._step_obs = step_obs
        self.steps = 0
        self.closed = False
        self.actions = []

    @property
    def instruction(self):
        return "INSTRUCTION"

    def reset(self):
        return self._obs

    def step(self, action):
        self.steps += 1
        self.actions.append(action)
        done = self._done_at is not None and self.steps >= self._done_at
        return StepResult(
            observation=self._step_obs,
            reward=self._reward_on_done if done else 0.0,
            done=done,
            info={},
        )

    def close(self):
        self.closed = True


def _fixed_counter(n):
    return lambda text: n


def _step(action):
    return f'<call name="step">{{"action": "{action}"}}</call>'


# --- run_env_episode ------------------------------------------------------


def test_env_episode_done_gives_reward():
    env = FakeEnv(done_at=1, reward_on_done=1.0)
    ep = run_env_episode(
        ScriptedModel([_step("go")]), "sys", env, max_turns=5, max_total_tokens=1000, count_tokens=_fixed_counter(1)
    )
    assert ep.stop_reason == "done"
    assert ep.env_done is True
    assert ep.reward == 1.0
    assert ep.tool_calls == 1
    assert env.closed is True


def test_env_episode_max_turns_no_reward():
    env = FakeEnv(done_at=None)
    ep = run_env_episode(
        ScriptedModel([_step("a"), _step("b")]), "sys", env, max_turns=2, max_total_tokens=1000, count_tokens=_fixed_counter(1)
    )
    assert ep.stop_reason == "max_turns"
    assert ep.reward == 0.0
    assert ep.env_done is False
    assert env.closed is True


def test_env_episode_token_budget_counts_observation():
    # reset obs (10) already; one turn output (10) + feedback (10) => stops next check.
    env = FakeEnv(done_at=None)
    ep = run_env_episode(
        ScriptedModel([_step("a"), _step("b")]), "sys", env, max_turns=10, max_total_tokens=25, count_tokens=_fixed_counter(10)
    )
    assert ep.stop_reason == "token_budget"
    # reset(10) + out(10) + feedback(10) = 30 >= 25, and only one step ran.
    assert env.steps == 1


def test_env_episode_format_error_then_retry():
    env = FakeEnv(done_at=1)
    ep = run_env_episode(
        ScriptedModel(["garbage no tag", _step("go")]),
        "sys",
        env,
        max_turns=5,
        max_total_tokens=1000,
        count_tokens=_fixed_counter(1),
    )
    assert ep.format_errors == 1
    assert ep.stop_reason == "done"
    assert ep.reward == 1.0


def test_env_episode_stops_on_done_no_extra_step():
    # Even with more scripted actions, the loop must stop the turn done fires.
    env = FakeEnv(done_at=1)
    run_env_episode(
        ScriptedModel([_step("a"), _step("b"), _step("c")]),
        "sys",
        env,
        max_turns=9,
        max_total_tokens=1000,
        count_tokens=_fixed_counter(1),
    )
    assert env.steps == 1  # did not step again after done


def test_env_episode_final_is_early_giveup():
    env = FakeEnv(done_at=None)
    ep = run_env_episode(
        ScriptedModel(["<final>giving up</final>"]),
        "sys",
        env,
        max_turns=5,
        max_total_tokens=1000,
        count_tokens=_fixed_counter(1),
    )
    assert ep.stop_reason == "final"
    assert ep.final_answer == "giving up"
    assert env.closed is True


# --- score_env_episode ----------------------------------------------------


def _episode(stop_reason, reward=0.0, env_done=False, format_errors=0, trace=None):
    return Episode(
        messages=[{"role": "user", "content": "last obs"}],
        final_answer=None,
        turns_used=3,
        tokens_used=100,
        stop_reason=stop_reason,
        format_errors=format_errors,
        trace=trace or [],
        reward=reward,
        env_done=env_done,
        max_turns=20,
        max_total_tokens=12287,
    )


def test_score_solved():
    score, fb = score_env_episode(_episode("done", reward=1.0, env_done=True))
    assert score == 1.0
    assert "success" in fb.lower()


def test_score_done_no_reward():
    score, _ = score_env_episode(_episode("done", reward=0.0, env_done=True))
    assert score == 0.0


def test_score_max_turns():
    score, fb = score_env_episode(_episode("max_turns"))
    assert score == 0.0
    assert "turns" in fb


def test_score_token_budget():
    score, _ = score_env_episode(_episode("token_budget"))
    assert score == 0.0


def test_score_truncated():
    score, _ = score_env_episode(_episode("truncated"))
    assert score == 0.0


# --- format-error gate ----------------------------------------------------


def test_score_format_errors_below_limit_still_credits_success():
    score, _ = score_env_episode(
        _episode("done", reward=1.0, env_done=True, format_errors=_FORMAT_ERROR_LIMIT - 1)
    )
    assert score == 1.0


def test_score_format_errors_at_limit_zeroes_a_solved_episode():
    """The gate must be checked BEFORE the success branch, or it never fires."""
    score, fb = score_env_episode(
        _episode("done", reward=1.0, env_done=True, format_errors=_FORMAT_ERROR_LIMIT)
    )
    assert score == 0.0
    assert "overrides the task outcome" in fb
    assert f"limit is {_FORMAT_ERROR_LIMIT}" in fb


# --- feedback surfaces the caps -------------------------------------------


def test_feedback_reports_budget_utilization_not_bare_counts():
    """The caps appear nowhere else the reflection LM can see."""
    _, fb = score_env_episode(_episode("done", reward=1.0, env_done=True))
    assert "turns=3/20" in fb
    assert "tokens=100/12287" in fb


def test_max_turns_feedback_names_the_cap():
    ep = _episode("max_turns")
    ep.turns_used = ep.max_turns
    _, fb = score_env_episode(ep)
    assert "HARD TURN CAP of 20" in fb
    assert "turns=20/20 (100%)" in fb


def test_token_budget_feedback_names_the_cap_and_reasoning_cost():
    ep = _episode("token_budget")
    ep.tokens_used = ep.max_total_tokens
    _, fb = score_env_episode(ep)
    assert "HARD TOKEN CAP of 12287" in fb
    assert "reasoning" in fb.lower()


def test_action_sequence_is_never_truncated():
    """Truncating hid the middle of the sequence, where inefficiency shows."""
    trace = [{"event": "step", "action": f"go to drawer {i}"} for i in range(1, 19)]
    _, fb = score_env_episode(_episode("max_turns", trace=trace))
    assert "more)" not in fb
    for i in range(1, 19):
        assert f"go to drawer {i}" in fb


# --- agentgym_datasets mapping --------------------------------------------


def test_alfworld_train_lookup(tmp_path, monkeypatch):
    mappings = [{"item_id": 7, "task_type": "pick", "task_id": "trial_X"}]
    p = tmp_path / "mappings_train.json"
    p.write_text(json.dumps(mappings))
    monkeypatch.setenv("AGENTGYM_ALFWORLD_MAPPINGS", str(p))
    assert _resolve_index("alfworld", "pick_trial_X", is_test=False, data_root=str(tmp_path)) == 7


def test_alfworld_test_tail():
    assert _resolve_index("alfworld", "alfworld_2420", is_test=True, data_root="/nope") == 2420


def test_textcraft_remap(tmp_path, monkeypatch):
    remap = {"textcraft_31": 32}
    p = tmp_path / "remap.json"
    p.write_text(json.dumps(remap))
    monkeypatch.setenv("AGENTGYM_TEXTCRAFT_REMAP", str(p))
    assert _resolve_index("textcraft", "textcraft_31", is_test=False, data_root=str(tmp_path)) == 32


def test_textcraft_test_tail():
    assert _resolve_index("textcraft", "textcraft_5", is_test=True, data_root="/nope") == 5


def test_load_agentgym_splits_seeded(tmp_path, monkeypatch):
    # Build a minimal AgentGym data tree and remap so no real files/servers are touched.
    (tmp_path / "data" / "train").mkdir(parents=True)
    (tmp_path / "data" / "test").mkdir(parents=True)
    train_ids = [f"textcraft_{i}" for i in range(10)]
    (tmp_path / "data" / "train" / "textcraft_train.json").write_text(
        json.dumps([{"item_id": i} for i in train_ids])
    )
    # Test ids must NOT collide with train ids here: this test is about the seeded
    # shuffle and split sizes, and the loader now drops train items that also
    # appear in test (see test_tool_loop_sciworld.py for that behavior). Reusing
    # ids 0-2 would silently shrink the train pool and confound this test.
    (tmp_path / "data" / "test" / "textcraft_test.json").write_text(
        json.dumps([{"item_id": f"textcraft_{i}"} for i in range(100, 103)])
    )
    remap = {i: n for n, i in enumerate(train_ids)}
    remap_path = tmp_path / "remap.json"
    remap_path.write_text(json.dumps(remap))
    monkeypatch.setenv("AGENTGYM_TEXTCRAFT_REMAP", str(remap_path))
    monkeypatch.setenv("TOOL_LOOP_ENV_SERVER", "http://127.0.0.1:36001")

    train, val, test = load_agentgym_splits(
        "textcraft", train_n=4, val_n=2, test_n=3, seed=0, data_root=str(tmp_path)
    )
    assert len(train) == 4
    assert len(val) == 2
    assert len(test) == 3
    # train/val disjoint (carved from the same shuffled train pool)
    train_items = {e.item_id for e in train}
    val_items = {e.item_id for e in val}
    assert train_items.isdisjoint(val_items)
    # reproducible with same seed
    train2, val2, _ = load_agentgym_splits(
        "textcraft", train_n=4, val_n=2, test_n=3, seed=0, data_root=str(tmp_path)
    )
    assert [e.item_id for e in train] == [e.item_id for e in train2]
    assert [e.item_id for e in val] == [e.item_id for e in val2]

    # Whole-split sentinels: test_n<=0 => all 3 test ids; train_n<=0 => all
    # remaining train after the val holdout; val_n<=0 => no val holdout.
    train_all, val_all, test_all = load_agentgym_splits(
        "textcraft", train_n=0, val_n=2, test_n=0, seed=0, data_root=str(tmp_path)
    )
    assert len(test_all) == 3  # entire test split
    assert len(train_all) == 8  # 10 train ids - 2 val holdout
    assert len(val_all) == 2
    _, val_none, _ = load_agentgym_splits(
        "textcraft", train_n=0, val_n=0, test_n=0, seed=0, data_root=str(tmp_path)
    )
    assert len(val_none) == 0  # no val holdout


# --- http_client / session (mocked requests) ------------------------------


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    """Stub for the trust_env=False session; records that proxies are ignored."""

    def __init__(self, response):
        self._response = response
        self.trust_env = False

    def post(self, *a, **k):
        return self._response

    def get(self, *a, **k):
        return self._response


def _patch_session(monkeypatch, response):
    monkeypatch.setattr(http_client, "_SESSION", _FakeSession(response))


def test_http_client_error_payload_raises(monkeypatch):
    _patch_session(monkeypatch, _FakeResponse({"error": "boom"}))
    with pytest.raises(EnvError):
        http_client.post("http://x", "/reset", {"id": 0})


def test_http_client_bad_status_raises(monkeypatch):
    _patch_session(monkeypatch, _FakeResponse({"observation": "x"}, 500))
    with pytest.raises(EnvError):
        http_client.post("http://x", "/step", {"id": 0})


def test_http_client_session_ignores_env_proxy():
    # trust_env=False is what makes localhost bypass http_proxy / glob no_proxy.
    assert http_client._session().trust_env is False


def test_textcraft_session_reset_step_close(monkeypatch):
    calls = []

    def fake_post(base, path, payload=None, *, timeout=300.0):
        calls.append((path, payload))
        if path == "/create":
            return {"id": 3, "observation": "", "done": False, "reward": 0}
        if path == "/reset":
            return {"id": 3, "observation": "GOAL: craft stick", "done": False, "reward": 0}
        if path == "/step":
            return {"observation": "crafted", "reward": 1.0, "done": True}
        if path == "/close":
            return {}
        raise AssertionError(path)

    monkeypatch.setattr(http_client, "post", fake_post)
    sess = TextCraftSession("http://127.0.0.1:36001", data_idx=32)
    obs = sess.reset()
    assert "craft stick" in obs
    result = sess.step("craft 1 stick using 2 planks")
    assert result.reward == 1.0
    assert result.done is True
    sess.close()
    paths = [c[0] for c in calls]
    assert paths == ["/create", "/reset", "/step", "/close"]
    # reset forwards data_idx; step forwards the raw action verbatim.
    assert calls[1][1]["data_idx"] == 32
    assert calls[2][1]["action"] == "craft 1 stick using 2 planks"


# --- profiles registry (reusable config contract) -------------------------


def test_profiles_registry_shape():
    from examples.tool_loop.profiles import PROFILES, get_profile

    assert set(PROFILES) >= {"guess", "textcraft", "alfworld"}
    for name, prof in PROFILES.items():
        assert prof.name == name
        assert prof.kind in ("answer", "env")
        # env tasks must know how to build a stateful session; answer tasks must not need one.
        assert (prof.make_session is not None) == (prof.kind == "env")
        # every profile declares the full set of tunable defaults.
        assert {"max_turns", "max_tokens", "max_workers", "train_n", "val_n", "test_n"} <= set(prof.defaults)
    assert get_profile("GUESS").name == "guess"  # case-insensitive


def test_profiles_alfworld_budget_is_tight():
    """AlfWorld runs on a deliberately tight budget so planning is what's tested.

    This used to assert AlfWorld got a *larger* budget than TextCraft (bigger
    observations, longer trajectories). That was inverted on purpose: an ample
    budget let the model brute-force by searching every container, which is the
    behavior the optimization is supposed to select against. 20 turns sits
    between the observed solved-episode median (11) and the failure median (~28).
    """
    from examples.tool_loop.profiles import PROFILES

    tc, aw = PROFILES["textcraft"].defaults, PROFILES["alfworld"].defaults
    assert aw["max_turns"] == 20
    assert aw["max_tokens"] == 12287
    assert aw["max_tokens"] < tc["max_tokens"]  # tight on purpose, not by omission
    # NOTE: this used to also assert aw["max_workers"] < tc["max_workers"], on the
    # theory that AlfWorld's missing /close route forced low concurrency. That was
    # wrong — instances are in-process and cheap, and accumulation scales with the
    # total episode count, not with concurrency. Both are now 15; see
    # test_tool_loop_parallel_eval.py::test_env_profiles_use_15_workers.


def test_profiles_unknown_task_raises():
    from examples.tool_loop.profiles import get_profile

    with pytest.raises(ValueError):
        get_profile("nope")
