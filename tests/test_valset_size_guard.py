"""Resuming a run_dir with a different valset size must fail loudly.

Observed failure: a smoke test wrote a 2-example state into the run dir a real
run later used. GEPA resumes whenever gepa_state.bin exists and only validated
frontier_type, so the real run silently continued on the old valset and logged
"Base program full valset score: 0.0 over 2 / 45 examples" — optimizing against
2 examples while reporting a 45-example configuration.
"""

import pytest

from gepa.core.state import ValsetEvaluation, initialize_gepa_state


class _Logger:
    def log(self, _msg):
        pass


def _valset_eval(n, with_objectives=False):
    return ValsetEvaluation(
        outputs_by_val_id={i: None for i in range(n)},
        scores_by_val_id={i: 0.0 for i in range(n)},
        objective_scores_by_val_id=({i: {"o": 0.0} for i in range(n)} if with_objectives else None),
        trajectories_by_val_id=None,
    )


def _seed_state(run_dir, n):
    """Create and persist a state built against an ``n``-example valset."""
    state = initialize_gepa_state(
        run_dir=str(run_dir),
        logger=_Logger(),
        seed_candidate={"current_candidate": "seed"},
        seed_valset_evaluation=_valset_eval(n),
        frontier_type="instance",
    )

    state.save(str(run_dir), use_cloudpickle=False)
    return state


def test_resume_with_a_different_valset_size_is_refused(tmp_path):
    _seed_state(tmp_path, 2)

    with pytest.raises(ValueError, match="Valset size mismatch"):
        initialize_gepa_state(
            run_dir=str(tmp_path),
            logger=_Logger(),
            seed_candidate={"current_candidate": "seed"},
            seed_valset_evaluation=_valset_eval(45),
            frontier_type="instance",
        )


def test_error_names_both_sizes(tmp_path):
    """The message must be actionable without opening the state file."""
    _seed_state(tmp_path, 2)

    with pytest.raises(ValueError) as excinfo:
        initialize_gepa_state(
            run_dir=str(tmp_path),
            logger=_Logger(),
            seed_candidate={"current_candidate": "seed"},
            seed_valset_evaluation=_valset_eval(45),
            frontier_type="instance",
        )
    msg = str(excinfo.value)
    assert "45 validation examples" in msg
    assert "built with 2" in msg
    assert "run_dir" in msg


def test_resume_with_the_same_valset_size_still_works(tmp_path):
    """The guard must not break legitimate resumes."""
    _seed_state(tmp_path, 8)

    state = initialize_gepa_state(
        run_dir=str(tmp_path),
        logger=_Logger(),
        seed_candidate={"current_candidate": "seed"},
        seed_valset_evaluation=_valset_eval(8),
        frontier_type="instance",
    )
    assert len(state.prog_candidate_val_subscores[0]) == 8


def test_fresh_run_dir_is_unaffected(tmp_path):
    """No prior state -> nothing to mismatch."""
    state = initialize_gepa_state(
        run_dir=str(tmp_path / "fresh"),
        logger=_Logger(),
        seed_candidate={"current_candidate": "seed"},
        seed_valset_evaluation=_valset_eval(45),
        frontier_type="instance",
    )
    assert len(state.prog_candidate_val_subscores[0]) == 45


def test_guard_uses_seed_subscores_not_pareto_front(tmp_path):
    """pareto_front_valset can hold ids from a later, differently sized eval.

    Keying the guard on it would report the wrong "loaded" size for any run dir
    that has seen two valset sizes, so the check must read the seed candidate's
    per-instance subscores instead.
    """
    state = _seed_state(tmp_path, 5)
    state.pareto_front_valset = {i: 0.0 for i in range(99)}
    state.program_at_pareto_front_valset = {i: {0} for i in range(99)}
    state.save(str(tmp_path), use_cloudpickle=False)

    # Still resumable at its true size (5), despite the 99-key pareto front.
    resumed = initialize_gepa_state(
        run_dir=str(tmp_path),
        logger=_Logger(),
        seed_candidate={"current_candidate": "seed"},
        seed_valset_evaluation=_valset_eval(5),
        frontier_type="instance",
    )
    assert len(resumed.prog_candidate_val_subscores[0]) == 5
