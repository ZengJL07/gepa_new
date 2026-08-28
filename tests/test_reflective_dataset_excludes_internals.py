"""Internal side_info keys must not reach the reflection prompt.

The eval server injects ``_budget`` (live eval counters) into every side_info.
Rendered into the reflection prompt, it made the prompt text differ between runs
whenever eval counts differed, so the reflection LM's cache always missed — and
that cascaded: a new candidate text changes the candidate hash, which misses the
fitness cache, which re-runs the episodes, which misses the solver cache. Replaying
an identical configuration therefore paid full price. These tests pin the
exclusion so caching stays keyed on real feedback.
"""

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.instruction_proposal import InstructionProposalSignature


def _side_info(used=60, max_evals=100, **extra):
    return {
        "score": 0.0,
        "input": "task text",
        "output": "",
        "stop_reason": "max_turns",
        "turns": 20,
        "tokens": 5000,
        "execution_feedback": "hit the HARD TURN CAP of 20 turns",
        "_budget": {
            "exhausted": False,
            "max_evals": max_evals,
            "used": used,
            "remaining_evals": max_evals - used,
        },
        **extra,
    }


def _reflective(side_info, component="c"):
    adapter = OptimizeAnythingAdapter(evaluator=lambda _c, _e: (0.0, {}))
    batch = EvaluationBatch(outputs=[None], scores=[0.0], trajectories=[side_info])
    return adapter.make_reflective_dataset({component: "seed"}, batch, [component])[component][0]


def test_budget_is_excluded():
    assert "_budget" not in _reflective(_side_info())


def test_all_underscore_keys_are_excluded():
    d = _reflective(_side_info(_gepa_transient_failure=True, _internal="x"))
    assert not [k for k in d if k.startswith("_")]


def test_real_feedback_is_retained():
    """The exclusion must not take actual feedback with it."""
    d = _reflective(_side_info())
    for key in ("score", "input", "output", "stop_reason", "turns", "tokens", "execution_feedback"):
        assert key in d, key


def test_component_specific_info_still_merges():
    d = _reflective(_side_info(c_specific_info={"hint": "use fewer turns"}), component="c")
    assert d["hint"] == "use fewer turns"


def test_scores_key_is_still_renamed():
    d = _reflective({"scores": {"acc": 1.0}, "execution_feedback": "fb"})
    assert d["Scores (Higher is Better)"] == {"acc": 1.0}
    assert "scores" not in d


def _render(used, max_evals):
    return InstructionProposalSignature.prompt_renderer(
        {
            "current_instruction_doc": "seed prompt",
            "dataset_with_feedback": [_reflective(_side_info(used, max_evals))],
            "prompt_template": None,
        }
    )


def test_prompt_is_stable_across_budget_caps():
    """Changing METRIC_CALLS must not invalidate the reflection LM cache."""
    assert _render(60, 100) == _render(60, 500)


def test_prompt_is_stable_across_eval_counts():
    """Nor must a one-off shift in how many evals have run so far."""
    assert _render(60, 100) == _render(61, 100)
