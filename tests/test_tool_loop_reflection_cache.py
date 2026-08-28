"""The reflection LM must sit behind a disk cache, or re-runs cannot replay.

GEPA's reflection LM is gepa.lm.LM calling litellm directly, so dspy's cache
never sees it. Left uncached it breaks replay for the WHOLE run, not just itself:
the reflector samples a different candidate text each time -> the candidate hash
changes -> the fitness cache misses -> the episodes re-run -> the solver cache
misses too. These tests pin the wiring that makes a second run free.
"""

import tempfile

import pytest

from examples.tool_loop.main import _configure_reflection_cache


@pytest.fixture
def _restore_litellm_cache():
    """Save/restore the global litellm cache so tests don't leak into each other."""
    import litellm

    prev_cache = getattr(litellm, "cache", None)
    prev_enabled = getattr(litellm, "enable_caching", None)
    yield
    litellm.cache = prev_cache
    if prev_enabled is None and hasattr(litellm, "disable_cache"):
        try:
            litellm.disable_cache()
        except Exception:  # noqa: BLE001 - best effort teardown
            pass


def test_configure_sets_a_global_disk_cache(_restore_litellm_cache):
    import litellm

    cache_dir = tempfile.mkdtemp(prefix="test-refl-cache-")
    litellm.cache = None
    _configure_reflection_cache(cache_dir)

    assert litellm.cache is not None, "global litellm.cache was not set"


def test_global_cache_intercepts_gepa_lm(_restore_litellm_cache):
    """gepa.lm.LM passes no caching= kwarg, so only a GLOBAL cache can catch it.

    Guards the core assumption: if a litellm upgrade stopped honoring the global
    cache for un-flagged calls, replay would silently start costing money again.
    """
    import litellm
    from litellm.types.utils import ModelResponse

    from gepa.lm import LM

    calls = {"n": 0}

    class _Counting(litellm.CustomLLM):
        def completion(self, *_args, **_kwargs):
            calls["n"] += 1
            return ModelResponse(
                id="x",
                choices=[
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": f"reply {calls['n']}", "role": "assistant"},
                    }
                ],
                created=0,
                model="counting/x",
                object="chat.completion",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

    prev_map = getattr(litellm, "custom_provider_map", [])
    litellm.custom_provider_map = [{"provider": "counting", "custom_handler": _Counting()}]
    try:
        _configure_reflection_cache(tempfile.mkdtemp(prefix="test-refl-hit-"))
        lm = LM("counting/model", temperature=0.0, max_tokens=16)

        first = lm("identical reflection prompt")
        assert calls["n"] == 1

        # Same prompt -> must be served from cache, byte-identical.
        second = lm("identical reflection prompt")
        assert calls["n"] == 1, f"cache miss: provider called {calls['n']}x"
        assert second == first

        # Different prompt -> must miss.
        lm("a different reflection prompt")
        assert calls["n"] == 2
    finally:
        litellm.custom_provider_map = prev_map


def test_batch_complete_is_also_cached(_restore_litellm_cache):
    """n_parallel > 1 reflection goes through batch_complete, not __call__."""
    import litellm
    from litellm.types.utils import ModelResponse

    from gepa.lm import LM

    calls = {"n": 0}

    class _Counting(litellm.CustomLLM):
        def completion(self, *_args, **_kwargs):
            calls["n"] += 1
            return ModelResponse(
                id="x",
                choices=[
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": f"batch {calls['n']}", "role": "assistant"},
                    }
                ],
                created=0,
                model="counting/x",
                object="chat.completion",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

    prev_map = getattr(litellm, "custom_provider_map", [])
    litellm.custom_provider_map = [{"provider": "counting", "custom_handler": _Counting()}]
    try:
        _configure_reflection_cache(tempfile.mkdtemp(prefix="test-refl-batch-"))
        lm = LM("counting/model", temperature=0.0, max_tokens=16)
        messages = [[{"role": "user", "content": "batch reflection prompt"}]]

        lm.batch_complete(messages)
        after_first = calls["n"]
        assert after_first >= 1

        lm.batch_complete(messages)
        assert calls["n"] == after_first, "batch_complete bypassed the cache"
    finally:
        litellm.custom_provider_map = prev_map


def test_configure_never_raises_on_a_bad_path(capsys):
    """A cache is an optimization; a broken path must not kill a paid-for run."""
    # A path under an existing *file* can never be created as a directory.
    with tempfile.NamedTemporaryFile() as f:
        _configure_reflection_cache(f"{f.name}/cache")

    assert "WARNING" in capsys.readouterr().out
