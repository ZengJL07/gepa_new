"""Tests for CapabilityTransferUCBSampling.

The strategy only reads three attributes off ``state``
(``program_candidates``, ``prog_candidate_val_subscores``,
``parent_program_for_candidate``), so we mock ``state`` with a
``SimpleNamespace`` rather than building a full GEPAState — mirroring
tests/test_batch_sampler.py.
"""

from types import SimpleNamespace

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.capability_transfer_sampling import CapabilityTransferUCBSampling


class FixedSelector:
    """CandidateSelector stub returning a preset parent idx (or a queue)."""

    def __init__(self, idx):
        self._idxs = list(idx) if isinstance(idx, (list, tuple)) else None
        self._idx = idx if self._idxs is None else None
        self._pos = 0

    def select_candidate_idx(self, state) -> int:
        if self._idxs is not None:
            val = self._idxs[self._pos % len(self._idxs)]
            self._pos += 1
            return val
        return self._idx


def make_state(candidates, subscores, parents):
    return SimpleNamespace(
        program_candidates=candidates,
        prog_candidate_val_subscores=subscores,
        parent_program_for_candidate=parents,
    )


def train_loader(k=6):
    return ListDataLoader([f"t{i}" for i in range(k)])


def test_emit_counts_nk_even_without_reconciled_candidate():
    # n_k must advance at emit time, before any candidate is accepted.
    strat = CapabilityTransferUCBSampling(n=1, minibatch_size=3, seed=0)
    state = make_state(
        candidates=[{"p": "seed"}],
        subscores=[{0: 0.0, 1: 0.0, 2: 1.0}],
        parents=[[None]],
    )
    tasks = strat.sample_tasks(state, FixedSelector(0), batch_sampler=None, trainset=train_loader())

    assert len(tasks) == 1
    assert len(tasks[0].minibatch_ids) == 3
    assert strat.total_emits == 1
    # Exactly the 3 emitted columns have n_k == 1.
    assert sum(strat.n_k.values()) == 3
    assert all(strat.n_k[k] == 1 for k in tasks[0].minibatch_ids)


def test_cold_start_bonus_and_determinism():
    # With an empty history every column has value 0 + cold_start_bonus, so
    # selection is a pure seeded tie-break. Two identically-seeded strategies
    # must return byte-identical minibatches.
    state = make_state([{"p": "s"}], [{0: 0.0, 1: 0.0}], [[None]])

    a = CapabilityTransferUCBSampling(n=1, minibatch_size=3, seed=123)
    b = CapabilityTransferUCBSampling(n=1, minibatch_size=3, seed=123)
    ta = a.sample_tasks(state, FixedSelector(0), None, train_loader())
    tb = b.sample_tasks(state, FixedSelector(0), None, train_loader())
    assert ta[0].minibatch_ids == tb[0].minibatch_ids

    # A different seed generally reorders the tie-break.
    c = CapabilityTransferUCBSampling(n=1, minibatch_size=3, seed=999)
    tc = c.sample_tasks(state, FixedSelector(0), None, train_loader())
    assert isinstance(tc[0].minibatch_ids, list)


def test_reconcile_updates_ab_with_row_mask():
    # Parent (idx 1) fails q0,q1 and solves q2. Child (idx 2) produced from
    # minibatch {t0,t1,t2} solves q0 (improved) but still fails q1.
    strat = CapabilityTransferUCBSampling(n=1, minibatch_size=3, seed=0)
    state = make_state(
        candidates=[{"p": "seed"}, {"p": "parent"}],
        subscores=[
            {0: 1.0, 1: 1.0, 2: 1.0},  # seed
            {0: 0.0, 1: 0.0, 2: 1.0},  # parent (idx 1)
        ],
        parents=[[None], [0]],
    )
    # Emit a task off parent 1; a 3-item loader forces the batch to be all
    # three columns. ListDataLoader keys columns by integer index (0,1,2).
    loader = ListDataLoader(["t0", "t1", "t2"])
    tasks = strat.sample_tasks(state, FixedSelector(1), None, loader)
    assert set(tasks[0].minibatch_ids) == {0, 1, 2}

    # Now a child appears (idx 2) with parent 1; reconcile on next call.
    state.program_candidates.append({"p": "child"})
    state.prog_candidate_val_subscores.append({0: 1.0, 1: 0.0, 2: 1.0})
    state.parent_program_for_candidate.append([1])
    strat._reconcile(state)

    # q0 was a masked-in failure that improved -> A += 1 for every column.
    # q1 was a masked-in failure that did NOT improve -> B += 1.
    # q2 was solved by parent -> masked out, no update.
    for k in (0, 1, 2):
        assert strat.A[(0, k)] == 1.0
        assert strat.B[(1, k)] == 1.0
        assert strat.A[(2, k)] == 0.0
        assert strat.B[(2, k)] == 0.0


def test_utility_prefers_columns_that_fixed_current_gaps():
    strat = CapabilityTransferUCBSampling(n=1, minibatch_size=1, cold_start_bonus=0.0, exploration_weight=0.0, seed=0)
    # Hand-craft history: column t0 has repeatedly fixed q0; t5 has not.
    for _ in range(5):
        strat.A[(0, "t0")] += 1
    strat.n_k["t0"] = 5
    strat.n_k["t5"] = 5
    strat.total_emits = 10

    # Parent fails q0 only.
    state = make_state([{"p": "s"}], [{0: 0.0, 1: 1.0}], [[None]])
    scores = strat._score_columns(state, 0, ["t0", "t5"])
    assert scores["t0"] > scores["t5"]


def test_parallel_tasks_get_distinct_columns():
    strat = CapabilityTransferUCBSampling(n=2, minibatch_size=3, seed=0)
    state = make_state([{"p": "s"}], [{0: 0.0}], [[None]])
    tasks = strat.sample_tasks(state, FixedSelector(0), None, train_loader(6))

    assert len(tasks) == 2
    mb0, mb1 = set(tasks[0].minibatch_ids), set(tasks[1].minibatch_ids)
    # 6 columns, 3 each -> disjoint.
    assert mb0.isdisjoint(mb1)


def test_parallel_reuses_columns_when_trainset_too_small():
    # 4 columns, 2 tasks of 3 -> cannot be disjoint; must not crash.
    strat = CapabilityTransferUCBSampling(n=2, minibatch_size=3, seed=0)
    state = make_state([{"p": "s"}], [{0: 0.0}], [[None]])
    tasks = strat.sample_tasks(state, FixedSelector(0), None, train_loader(4))
    assert len(tasks) == 2
    assert all(len(t.minibatch_ids) == 3 for t in tasks)


def test_empty_trainset_raises():
    strat = CapabilityTransferUCBSampling(n=1, minibatch_size=3, seed=0)
    state = make_state([{"p": "s"}], [{0: 0.0}], [[None]])
    with pytest.raises(ValueError):
        strat.sample_tasks(state, FixedSelector(0), None, ListDataLoader([]))


class StubProposal:
    """Minimal CandidateProposal stand-in for observe_proposals."""

    def __init__(self, subsample_indices, before, after):
        self.subsample_indices = subsample_indices
        self.subsample_scores_before = before
        self.subsample_scores_after = after


def test_observe_proposals_counts_obs_and_fix():
    strat = CapabilityTransferUCBSampling(minibatch_size=1, tau=0.5, seed=0)
    # t0: parent failed (0.0) then child passed (1.0) -> a fix.
    # t1: parent already solved (1.0) -> obs only, never a fix.
    # t2: parent failed and child still fails -> obs only (stuck).
    strat.observe_proposals(
        [StubProposal(["t0", "t1", "t2"], before=[0.0, 1.0, 0.0], after=[1.0, 1.0, 0.0])]
    )
    assert strat.obs["t0"] == 1.0 and strat.fix["t0"] == 1.0
    assert strat.obs["t1"] == 1.0 and strat.fix["t1"] == 0.0
    assert strat.obs["t2"] == 1.0 and strat.fix["t2"] == 0.0
    # U ordering: fixable > (ceiling / stuck).
    assert strat._usability("t0") > strat._usability("t1")
    assert strat._usability("t0") > strat._usability("t2")


def test_observe_proposals_ignores_misaligned_or_missing():
    strat = CapabilityTransferUCBSampling(seed=0)
    strat.observe_proposals([StubProposal(["t0", "t1"], before=[0.0], after=[1.0])])  # length mismatch
    strat.observe_proposals([StubProposal(None, None, None)])  # missing fields
    assert sum(strat.obs.values()) == 0.0


def test_usability_gate_downweights_unfixable_column():
    # Two columns identically relevant to the parent's gap q0, but t_bad has a
    # long history of being sampled without ever being fixed (stuck/too-hard),
    # while t_good has been fixed. The gate must rank t_good above t_bad.
    strat = CapabilityTransferUCBSampling(
        minibatch_size=1, cold_start_bonus=0.0, exploration_weight=0.0, usability_weight=1.0, seed=0
    )
    for _ in range(5):
        strat.A[(0, "t_good")] += 1
        strat.A[(0, "t_bad")] += 1  # identical relevance to q0
    # Usability histories diverge.
    strat.obs["t_good"] = 5.0
    strat.fix["t_good"] = 5.0
    strat.obs["t_bad"] = 20.0
    strat.fix["t_bad"] = 0.0

    state = make_state([{"p": "s"}], [{0: 0.0}], [[None]])
    scores = strat._score_columns(state, 0, ["t_good", "t_bad"])
    assert scores["t_good"] > scores["t_bad"]


def test_usability_weight_zero_disables_gate():
    # With gamma == 0 the gate is 1.0, so identical relevance -> identical score
    # regardless of usability history.
    strat = CapabilityTransferUCBSampling(
        minibatch_size=1, cold_start_bonus=0.0, exploration_weight=0.0, usability_weight=0.0, seed=0
    )
    for _ in range(5):
        strat.A[(0, "t_good")] += 1
        strat.A[(0, "t_bad")] += 1
    strat.obs["t_good"] = 5.0
    strat.fix["t_good"] = 5.0
    strat.obs["t_bad"] = 20.0
    strat.fix["t_bad"] = 0.0

    state = make_state([{"p": "s"}], [{0: 0.0}], [[None]])
    scores = strat._score_columns(state, 0, ["t_good", "t_bad"])
    assert scores["t_good"] == scores["t_bad"]


def test_merge_candidates_are_not_reconciled():
    # A two-parent (merge) candidate must be skipped by reconciliation.
    strat = CapabilityTransferUCBSampling(n=1, minibatch_size=3, seed=0)
    state = make_state(
        candidates=[{"p": "seed"}, {"p": "a"}, {"p": "b"}],
        subscores=[{0: 0.0}, {0: 0.0}, {0: 0.0}],
        parents=[[None], [0], [0]],
    )
    loader = ListDataLoader(["t0", "t1", "t2"])
    strat.sample_tasks(state, FixedSelector(0), None, loader)  # emit off parent 0
    # A merge child of parents [1, 2] appears.
    state.program_candidates.append({"p": "merged"})
    state.prog_candidate_val_subscores.append({0: 1.0})
    state.parent_program_for_candidate.append([1, 2])
    strat._reconcile(state)
    # No A/B entries: the pending edge for parent 0 stays unconsumed.
    assert sum(strat.A.values()) == 0.0
    assert sum(strat.B.values()) == 0.0
