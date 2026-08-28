"""evaluate_on_dataset must run episodes concurrently, without changing results.

The held-out test passes used to be a serial loop, making them the slowest part
of a run: TEST_SIZE=500 x up to 20 LLM round-trips per episode, all sequential,
while the search phase was already parallel. These tests pin the concurrency and
the invariants that make it safe.
"""

import threading
import time

import pytest

from examples.tool_loop import main as tl
from examples.tool_loop.task_env import Episode


class _Ex:
    def __init__(self, i):
        self.target = i
        self.item_id = f"item_{i}"
        self.input = f"in {i}"

    def with_inputs(self, *_names):
        return self


@pytest.fixture
def tracker(monkeypatch):
    """Replace _run_one with a latency stub that records peak concurrency."""
    state = {"live": 0, "peak": 0, "calls": 0}
    lock = threading.Lock()

    def fake_run_one(_prompt, example, latency=0.02):
        with lock:
            state["live"] += 1
            state["calls"] += 1
            state["peak"] = max(state["peak"], state["live"])
        try:
            time.sleep(latency)
            i = example.target
            episode = Episode(
                messages=[],
                final_answer=str(i),
                turns_used=1,
                tokens_used=1,
                stop_reason="final",
                max_turns=20,
                max_total_tokens=12287,
            )
            return episode, (1.0 if i % 2 == 0 else 0.0), f"fb {i}"
        finally:
            with lock:
                state["live"] -= 1

    monkeypatch.setattr(tl, "_run_one", fake_run_one)
    return state


def test_parallel_matches_serial_score(tracker):
    data = [_Ex(i) for i in range(20)]
    serial = tl.evaluate_on_dataset("p", data, max_workers=1)
    parallel = tl.evaluate_on_dataset("p", data, max_workers=8)
    assert serial == parallel == 0.5


def test_episodes_actually_run_concurrently(tracker):
    tl.evaluate_on_dataset("p", [_Ex(i) for i in range(20)], max_workers=8)
    assert tracker["peak"] > 1, "evaluation ran serially"


def test_concurrency_never_exceeds_max_workers(tracker):
    tl.evaluate_on_dataset("p", [_Ex(i) for i in range(30)], max_workers=5)
    assert tracker["peak"] <= 5, tracker["peak"]


def test_concurrency_never_exceeds_dataset_size(tracker):
    """A 3-example dataset must not spin up 15 threads."""
    tl.evaluate_on_dataset("p", [_Ex(i) for i in range(3)], max_workers=15)
    assert tracker["peak"] <= 3, tracker["peak"]


def test_hook_sees_every_example_exactly_once(tracker):
    data = [_Ex(i) for i in range(25)]
    seen = []
    tl.evaluate_on_dataset("p", data, on_episode=lambda i, *_a: seen.append(i), max_workers=8)
    assert sorted(seen) == list(range(25))


def test_hook_is_never_called_concurrently(tracker):
    """Callers pass plain appenders, so the hook must be serialized for them."""
    overlaps = []
    depth = [0]
    lock = threading.Lock()

    def hook(_i, *_a):
        with lock:
            depth[0] += 1
            if depth[0] > 1:
                overlaps.append(True)
        time.sleep(0.001)
        with lock:
            depth[0] -= 1

    tl.evaluate_on_dataset("p", [_Ex(i) for i in range(20)], on_episode=hook, max_workers=8)
    assert not overlaps, "hook was entered concurrently"


def test_exceptions_propagate_rather_than_scoring_zero(tracker, monkeypatch):
    """A crash in the final eval is a failure, not a wrong answer."""
    original = tl._run_one

    def boom(prompt, example):
        if example.target == 3:
            raise RuntimeError("episode blew up")
        return original(prompt, example)

    monkeypatch.setattr(tl, "_run_one", boom)
    with pytest.raises(RuntimeError, match="episode blew up"):
        tl.evaluate_on_dataset("p", [_Ex(i) for i in range(10)], max_workers=4)


def test_empty_dataset_short_circuits(tracker):
    assert tl.evaluate_on_dataset("p", []) == 0.0
    assert tracker["calls"] == 0


def test_worker_count_resolution_order(tracker, monkeypatch):
    """TOOL_LOOP_MAX_WORKERS wins over AIME_MAX_WORKERS, which wins over profile."""
    monkeypatch.setenv("AIME_MAX_WORKERS", "6")
    monkeypatch.setenv("TOOL_LOOP_MAX_WORKERS", "2")
    assert tl._resolve_max_workers() == 2

    monkeypatch.delenv("TOOL_LOOP_MAX_WORKERS")
    assert tl._resolve_max_workers() == 6

    monkeypatch.delenv("AIME_MAX_WORKERS")
    assert tl._resolve_max_workers() == tl._PROFILE.defaults["max_workers"]


def test_env_profiles_use_15_workers():
    """AlfWorld's old value of 3 throttled the run for a reason that did not hold."""
    from examples.tool_loop.profiles import PROFILES

    assert PROFILES["alfworld"].defaults["max_workers"] == 15
    assert PROFILES["textcraft"].defaults["max_workers"] == 15
