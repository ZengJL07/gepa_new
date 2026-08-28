"""ScienceWorld adapter: score normalization, session lifecycle, split hygiene.

Facts established live against the AgentGym server on this host, pinned here
because the adapter depends on them:

1. ``score`` is CUMULATIVE progress on a 0-100 scale; ``reward`` is the per-step
   delta. Reporting the delta would read a winning move as failure.
2. Replaying an expert trajectory reached ``score=100`` with ``done=True``, so
   "fully solved" really is ``normalized reward >= 1.0``.
3. ``item_id`` tail == server ``data_idx`` for all 2120 train items, so no remap
   table is needed (unlike TextCraft).
"""

import pytest

from examples.tool_loop.agentgym_datasets import _resolve_index
from examples.tool_loop.envs.sciworld import SciWorldSession, _normalize_score


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, 0.0),
        (3, 0.03),
        (67, 0.67),
        (99, 0.99),
        (100, 1.0),
        (150, 1.0),  # clamped: never exceed a full solve
        (-100, 0.0),  # ScienceWorld's "unrecoverable" sentinel
        ("nonsense", 0.0),
        (None, 0.0),
    ],
)
def test_score_normalization(raw, expected):
    assert _normalize_score(raw) == pytest.approx(expected)


def test_only_a_full_score_counts_as_solved():
    """The shared scorer tests reward >= 1.0, so 99/100 must stay below it."""
    assert _normalize_score(99) < 1.0
    assert _normalize_score(100) >= 1.0


class _StubHTTP:
    """Records posts and replays canned responses, so no server is needed."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def post(self, base, path, payload=None, *, timeout=None):
        self.calls.append((path, payload))
        return self.responses[path]


@pytest.fixture
def stub(monkeypatch):
    responses = {
        "/create": {"id": 7},
        "/reset": {
            "task_name": "find-plant",
            "task_description": "Your task is to find a(n) plant.",
            "observation": "This room is called the workshop.",
            "score": 0,
            "reward": 0,
            "done": False,
        },
        "/step": {"observation": "You move.", "reward": 6, "score": 9, "done": False},
        "/close": True,
    }
    s = _StubHTTP(responses)
    monkeypatch.setattr("examples.tool_loop.envs.sciworld.http_client", s)
    return s


def test_reset_prepends_the_task_description(stub):
    """The goal lives in task_description, NOT the observation. Dropping it would
    leave the model with no idea what it is being asked to do."""
    obs = SciWorldSession("http://x", 3491).reset()
    assert obs.startswith("Your task is to find a(n) plant.")
    assert "This room is called the workshop." in obs


def test_reset_adopts_the_server_session_id(stub):
    sess = SciWorldSession("http://x", 3491, session_id=0)
    sess.reset()
    reset_payload = dict(stub.calls)["/reset"]
    assert reset_payload == {"id": 7, "data_idx": 3491}


def test_step_reports_cumulative_score_not_the_delta(stub):
    """/step returns reward=6 (delta) and score=9 (cumulative) -> expect 0.09."""
    sess = SciWorldSession("http://x", 3491)
    sess.reset()
    result = sess.step("go to kitchen")
    assert result.reward == pytest.approx(0.09)
    assert result.info["raw_score"] == 9
    assert result.info["step_reward"] == 6


def test_close_releases_the_session(stub):
    """ScienceWorld HAS a /close route, unlike AlfWorld — use it."""
    sess = SciWorldSession("http://x", 3491)
    sess.reset()
    sess.close()
    assert ("/close", {"id": 7}) in stub.calls


def test_close_before_reset_is_a_noop(stub):
    SciWorldSession("http://x", 3491).close()
    assert stub.calls == []


def test_close_is_idempotent(stub):
    sess = SciWorldSession("http://x", 3491)
    sess.reset()
    sess.close()
    sess.close()
    assert sum(1 for path, _ in stub.calls if path == "/close") == 1


def test_close_never_raises(monkeypatch):
    """Teardown must not mask a real failure in the episode itself."""
    from examples.tool_loop.envs.base import EnvError

    class Boom(_StubHTTP):
        def post(self, base, path, payload=None, *, timeout=None):
            if path == "/close":
                raise EnvError("server gone")
            return super().post(base, path, payload, timeout=timeout)

    s = Boom({"/create": {"id": 1}, "/reset": {"observation": "o", "score": 0}, "/step": {}})
    monkeypatch.setattr("examples.tool_loop.envs.sciworld.http_client", s)
    sess = SciWorldSession("http://x", 1)
    sess.reset()
    sess.close()  # must not raise


def test_index_is_tail_int_for_both_splits():
    for is_test in (False, True):
        got = _resolve_index("sciworld", "sciworld_4632", is_test=is_test, data_root="/nonexistent")
        assert got == 4632


def test_unknown_env_name_is_rejected():
    with pytest.raises(ValueError, match="sciworld"):
        _resolve_index("nope", "nope_1", is_test=False, data_root="/nonexistent")


def test_profile_is_registered_and_reuses_the_shared_scorer():
    from examples.tool_loop.agentgym_scoring import score_env_episode
    from examples.tool_loop.profiles import get_profile

    prof = get_profile("sciworld")
    assert prof.kind == "env"
    assert prof.make_session is not None
    # Same scorer as AlfWorld/TextCraft: normalization is what makes this valid.
    ep_score, _ = prof.scorer(_solved_episode(), None)
    assert ep_score == 1.0
    assert score_env_episode(_solved_episode(), None)[0] == 1.0


def test_train_pool_excludes_ids_present_in_test(tmp_path):
    """ScienceWorld's official train/test files share 61 item_ids. Since val is
    carved from train, leaving them in means optimizing on items that are later
    scored — contamination, not a sampling artifact."""
    import json

    from examples.tool_loop.agentgym_datasets import load_agentgym_splits

    root = tmp_path
    (root / "data" / "train").mkdir(parents=True)
    (root / "data" / "test").mkdir(parents=True)
    # ids 0..9 in train; 5..9 also appear in test.
    with open(root / "data" / "train" / "sciworld_train.json", "w") as f:
        json.dump([{"item_id": f"sciworld_{i}"} for i in range(10)], f)
    with open(root / "data" / "test" / "sciworld_test.json", "w") as f:
        json.dump([{"item_id": f"sciworld_{i}"} for i in range(5, 10)], f)

    train, val, test = load_agentgym_splits(
        "sciworld", train_n=0, val_n=2, test_n=0, seed=1, data_root=str(root)
    )
    drawn = {e.item_id for e in train} | {e.item_id for e in val}
    assert drawn == {f"sciworld_{i}" for i in range(5)}
    assert not drawn & {e.item_id for e in test}


def _solved_episode():
    from examples.tool_loop.task_env import Episode

    return Episode(
        messages=[],
        final_answer=None,
        turns_used=5,
        tokens_used=100,
        stop_reason="done",
        reward=1.0,
        env_done=True,
        max_turns=30,
        max_total_tokens=16000,
    )
