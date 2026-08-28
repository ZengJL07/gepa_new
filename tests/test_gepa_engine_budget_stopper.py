"""The gepa engine must end a run on the eval ledger, not on BudgetExhausted.

Two eval counters exist and they drift: core's MaxMetricCallsStopper reads
GEPAState.total_num_evals (cache misses, updated at iteration boundaries) while
the EvalServer's BudgetTracker is what actually raises. Core could therefore
start an iteration it could not fund, and BudgetExhausted surfaced from inside a
parallel valset eval — discarding that iteration's paid-for evals and printing a
traceback at what is really a normal end-of-budget stop.
"""

import itertools
import tempfile

import pytest

from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig
from gepa.oa.budget import BudgetTracker
from gepa.oa.engines.gepa import _install_budget_stopper, _ServerBudgetStopper
from gepa.optimize_anything import optimize_anything


class _Ex:
    def __init__(self, i):
        self.i = i
        self.input = f"item {i}"

    def with_inputs(self, *_names):
        return self


class _ReflectLM:
    """Stub reflection LM: always proposes a new, distinct candidate."""

    total_cost = 0.0

    def __init__(self):
        self.n = 0

    def __call__(self, prompt=None, **_kw):
        self.n += 1
        return f"```\nvariant {self.n}\n```"


def _run(budget, n_val, n_train=12, minibatch=3, capsys=None):
    """Run a full optimization whose budget is too small to finish the search."""
    counter = itertools.count()

    def evaluate(_candidate, _example):
        # Unique per call so the adapter cache never hides an eval from the
        # server's ledger — that's what makes the two counters diverge.
        n = next(counter)
        return (n % 5) / 5.0, {"score": (n % 5) / 5.0, "execution_feedback": f"call {n}"}

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=tempfile.mkdtemp(prefix="test-budget-"),
            seed=42,
            max_metric_calls=budget,
            track_best_outputs=True,
            parallel=True,
            max_workers=4,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(reflection_lm=_ReflectLM(), reflection_minibatch_size=minibatch),
    )
    result = optimize_anything(
        seed_candidate="seed prompt",
        evaluator=evaluate,
        dataset=[_Ex(i) for i in range(n_train)],
        valset=[_Ex(100 + i) for i in range(n_val)],
        config=config,
    )
    return result


# (budget, valset) pairs that all used to raise BudgetExhausted mid-iteration.
@pytest.mark.parametrize(
    ("budget", "n_val"),
    [(100, 45), (100, 8), (60, 45), (30, 10), (14, 10)],
)
def test_budget_exhaustion_never_surfaces(budget, n_val, capsys):
    result = _run(budget, n_val)

    # The run completes and yields a usable result.
    assert result is not None
    assert result.best_candidate

    # Nothing was logged about an exhausted budget or a crashed iteration.
    out = capsys.readouterr().out
    assert "Eval budget exhausted" not in out
    assert "Exception during optimization" not in out


def test_budget_is_actually_respected():
    """The stopper must not overshoot the cap it is protecting."""
    budget = 100
    result = _run(budget, n_val=8)
    used = result.total_metric_calls
    assert used is not None
    # Core stops at iteration boundaries, so a small overshoot is by design;
    # what must not happen is blowing far past the cap.
    assert used <= budget + 8, used


def test_stopper_reserves_one_iteration():
    """Stop while a full iteration (minibatch + valset) can no longer be funded."""
    tracker = BudgetTracker(max_evals=10)
    stopper = _ServerBudgetStopper(tracker, reserve=4)

    assert stopper(None) is False  # 10 remaining > 4
    for _ in range(6):
        tracker.record(0.0)
    assert tracker.remaining == 4
    assert stopper(None) is True  # exactly at the reserve -> stop


def test_stopper_is_noop_when_budget_unlimited():
    stopper = _ServerBudgetStopper(BudgetTracker(max_evals=None), reserve=5)
    assert stopper(None) is False


def test_install_composes_with_existing_stoppers():
    """The stopper is appended, never replacing a user's own stop condition."""
    sentinel = object()

    cfg = GEPAConfig(engine=EngineConfig(max_metric_calls=50))
    cfg.stop_callbacks = sentinel  # a single (non-sequence) stopper
    _install_budget_stopper(cfg, BudgetTracker(max_evals=50), reserve=3)
    assert cfg.stop_callbacks[0] is sentinel
    assert isinstance(cfg.stop_callbacks[-1], _ServerBudgetStopper)

    cfg2 = GEPAConfig(engine=EngineConfig(max_metric_calls=50))
    cfg2.stop_callbacks = [sentinel]  # a list of stoppers
    _install_budget_stopper(cfg2, BudgetTracker(max_evals=50), reserve=3)
    assert cfg2.stop_callbacks[0] is sentinel
    assert isinstance(cfg2.stop_callbacks[-1], _ServerBudgetStopper)


def test_install_is_noop_for_unlimited_budget():
    cfg = GEPAConfig(engine=EngineConfig(max_metric_calls=None))
    _install_budget_stopper(cfg, BudgetTracker(max_evals=None), reserve=3)
    assert cfg.stop_callbacks is None
