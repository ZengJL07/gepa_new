"""An env-server hiccup must cost one episode, not the whole run.

Observed failure: mid-run, AlfWorld /step raised
ConnectionResetError(104, 'Connection reset by peer'). It propagated out of the
worker thread, through the adapter, out of core.engine.run() (raise_on_exception
defaults to True), and killed the process — discarding that iteration's paid-for
evals and writing no summary at all.

Root causes, all covered here:
  1. requests' default pool_maxsize=10 < our 15 episode workers, so surplus
     threads churned fresh sockets instead of reusing pooled ones.
  2. uvicorn closes idle keep-alive connections after 5s, but an episode can
     leave one idle far longer while waiting on the solver LLM. The client only
     learns the socket is dead when it writes -> ConnectionReset.
  3. Nothing contained the failure, and nothing salvaged the search results.
"""

import json
import os
import tempfile

import pytest
import requests

from examples.tool_loop import main as tl
from examples.tool_loop.envs.base import EnvError
from examples.tool_loop.task_env import Episode


class _Ex:
    def __init__(self, i):
        self.item_id = f"item_{i}"
        self.env_index = i
        self.input = f"in {i}"

    def with_inputs(self, *_names):
        return self


def _ok_episode():
    return Episode(
        messages=[],
        final_answer="x",
        turns_used=2,
        tokens_used=10,
        stop_reason="done",
        reward=1.0,
        env_done=True,
        max_turns=20,
        max_total_tokens=12287,
    )


# --- 1. HTTP client hardening ------------------------------------------------


def test_pool_is_at_least_as_large_as_episode_concurrency():
    """pool_maxsize < max_workers makes the pool thrash under load."""
    from examples.tool_loop.envs import http_client
    from examples.tool_loop.profiles import PROFILES

    adapter = http_client._session().get_adapter("http://127.0.0.1:36002")
    needed = max(p.defaults["max_workers"] for p in PROFILES.values())
    assert adapter._pool_maxsize >= needed, (adapter._pool_maxsize, needed)


def test_dropped_connections_are_retried():
    """Retrying POST is safe: urllib3 only retries when no response was read."""
    from examples.tool_loop.envs import http_client

    retries = http_client._session().get_adapter("http://127.0.0.1:36002").max_retries
    assert retries.total >= 3
    assert retries.connect >= 3
    assert "POST" in (retries.allowed_methods or set())
    assert retries.backoff_factor > 0, "no backoff means retries hammer a busy server"


# --- 2. Failure containment --------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.ConnectionError(
            "('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))"
        ),
        requests.exceptions.ReadTimeout("timed out"),
        EnvError("/step -> server error: The task with environment 7 has finished."),
    ],
)
def test_transport_failure_scores_zero_instead_of_killing_the_run(monkeypatch, exc):
    def flaky(_prompt, example):
        if example.env_index == 2:
            raise exc
        return _ok_episode(), 1.0, "ok"

    monkeypatch.setattr(tl, "_run_one", flaky)
    data = [_Ex(i) for i in range(5)]

    score = tl.evaluate_on_dataset("p", data, max_workers=3)

    assert score == pytest.approx(4 / 5), score


def test_env_error_episode_is_scored_and_explained(monkeypatch):
    """The 0 must carry a reason, so the reflector is not told the prompt failed."""
    monkeypatch.setattr(
        tl,
        "_run_one",
        lambda _p, _e: (_ for _ in ()).throw(requests.exceptions.ConnectionError("reset")),
    )
    episode, score, feedback = tl._run_one_resilient("p", _Ex(0))

    assert score == 0.0
    assert episode.stop_reason == "env_error"
    assert "environment server failed" in feedback


def test_real_bugs_still_propagate(monkeypatch):
    """Swallowing a genuine bug would corrupt the search with no warning."""

    def buggy(_prompt, _example):
        raise ValueError("a genuine bug in scoring")

    monkeypatch.setattr(tl, "_run_one", buggy)
    with pytest.raises(ValueError, match="genuine bug"):
        tl.evaluate_on_dataset("p", [_Ex(i) for i in range(4)], max_workers=2)


# --- 3. Crash salvage --------------------------------------------------------


def test_best_prompt_is_salvaged_from_persisted_candidates():
    """GEPA writes candidates.json each iteration; a crash must not waste them."""
    run_dir = tempfile.mkdtemp(prefix="test-salvage-")
    with open(os.path.join(run_dir, "candidates.json"), "w") as f:
        json.dump(
            [{"current_candidate": "seed"}, {"current_candidate": "improved in iter 2"}],
            f,
        )

    assert tl._salvage_best_prompt(run_dir, "fallback") == "improved in iter 2"


def test_salvage_falls_back_when_nothing_usable_on_disk():
    empty = tempfile.mkdtemp(prefix="test-salvage-empty-")
    assert tl._salvage_best_prompt(empty, "fallback") == "fallback"

    with open(os.path.join(empty, "candidates.json"), "w") as f:
        f.write("{ not valid json")
    assert tl._salvage_best_prompt(empty, "fallback") == "fallback"

    with open(os.path.join(empty, "candidates.json"), "w") as f:
        json.dump([], f)
    assert tl._salvage_best_prompt(empty, "fallback") == "fallback"
