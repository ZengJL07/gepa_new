# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the APEX dynamic data selection strategies.

Covers Algorithm 1 / Algorithm 2 of *APEX: Automated Prompt Engineering eXpert
with Dynamic Data Selection* (arXiv:2606.11459v1) as implemented in
``gepa.strategies.apex_*``.

State is mocked with ``SimpleNamespace`` rather than a full GEPAState, mirroring
tests/test_capability_transfer_sampling.py.
"""

from types import SimpleNamespace

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.acceptance import AlwaysAcceptance, StrictImprovementAcceptance
from gepa.strategies.apex_candidate_selector import ApexCurrentBestSelector
from gepa.strategies.apex_eval_policy import ApexRankSensitivePolicy
from gepa.strategies.apex_reflection import (
    CRITIQUE_PROMPT_TEMPLATE,
    MUTATION_PROMPT_TEMPLATE,
    ApexTwoStepReflection,
    render_error_case,
)
from gepa.strategies.apex_sampling import ApexDynamicSampling
from gepa.strategies.apex_stratification import (
    HistoryEntry,
    HistoryView,
    RejectedHistoryTracker,
    build_history,
    stratify,
    tier_of,
)


class FixedSelector:
    """CandidateSelector stub returning a preset parent idx."""

    def __init__(self, idx: int):
        self.idx = idx

    def select_candidate_idx(self, state) -> int:
        return self.idx


def make_state(subscores, n_candidates=None, i=0):
    n = n_candidates if n_candidates is not None else len(subscores)
    return SimpleNamespace(
        program_candidates=[{"sys": f"c{k}"} for k in range(n)],
        prog_candidate_val_subscores=subscores,
        parent_program_for_candidate=[[None]] + [[0]] * (n - 1),
        i=i,
        get_program_average_val_subset=lambda idx: (
            (sum(subscores[idx].values()) / len(subscores[idx]), len(subscores[idx]))
            if subscores[idx]
            else (float("-inf"), 0)
        ),
    )


def loader(k=10):
    return ListDataLoader([f"x{i}" for i in range(k)])


# ----------------------------------------------------------------------
# Section 4.1 / Algorithm 2: tiers and buckets
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("local_history", "expected"),
    [
        ([1, 1, 1], "E"),
        ([0, 0], "H"),
        ([1, 0, 1], "M"),
        ([0, 1], "M"),
        ([], None),
    ],
)
def test_tier_assignment_follows_eq7(local_history, expected):
    assert tier_of(local_history) == expected


def test_local_history_window_is_per_example():
    # Eq. 6: R_i is the last k outcomes among the prompts that actually
    # evaluated x_i -- prompts that skipped it are invisible to its window.
    history = HistoryView(
        [
            HistoryEntry(0, {0: 1, 1: 0}),
            HistoryEntry(1, {0: 0}),
            HistoryEntry(2, {0: 0}),
            HistoryEntry(3, {0: 1, 1: 1}),
        ]
    )
    assert history.local_history(0, 2) == [0, 1]
    # id 1 was scored only by entries 0 and 3; entries 1-2 do not consume window.
    assert history.local_history(1, 5) == [0, 1]


def test_stratify_builds_nine_buckets():
    history = HistoryView(
        [
            HistoryEntry(0, {0: 1, 1: 0, 2: 1, 3: 0}),
            HistoryEntry(1, {0: 0, 1: 0}),
        ]
    )
    strat = stratify([0, 1, 2, 3], history, current_key=1, lookback=5)

    assert strat.tiers == {0: "M", 1: "H", 2: "E", 3: "H"}
    assert strat.bucket("M", 0) == [0]
    assert strat.bucket("H", 0) == [1]
    # Never evaluated by candidate 1 -> s = None.
    assert strat.bucket("E", None) == [2]
    assert strat.bucket("H", None) == [3]


def test_none_bucket_means_p_curr_has_no_recorded_outcome():
    # Algorithm 2 line 10: s is looked up in H and "yields None if uneval". That
    # is what makes D_req = B[M, None] (line 8) a set of volatile *unknowns*.
    # An earlier version instead forced None for any id outside the previous
    # iteration's D_eval; once the engine began re-scoring P_curr each iteration
    # its coverage outgrew any single D_eval, and ids P_curr had already answered
    # well ended up in D_req -- filling the "unbiased baseline" with incumbent
    # strengths so no challenger could win line 16.
    history = HistoryView([HistoryEntry(0, {0: 1, 1: 0, 2: 0}), HistoryEntry(1, {0: 0, 1: 1})])

    strat = stratify([0, 1, 2], history, current_key=1, lookback=5)

    # P_curr (candidate 1) scored ids 0 and 1, so those carry a known outcome.
    assert strat.bucket("M", 0) == [0]
    assert strat.bucket("M", 1) == [1]
    # Id 2 was never scored by candidate 1 -> None, regardless of candidate 0.
    assert strat.bucket("H", None) == [2]
    # And nothing P_curr knows about may sit in a None bucket.
    known = set(history.entries[1].outcomes)
    for tier in ("E", "H", "M"):
        assert not (set(strat.bucket(tier, None)) & known)


# ----------------------------------------------------------------------
# Algorithm 1 line 19: what enters H, and when
# ----------------------------------------------------------------------


def test_pending_proposals_do_not_affect_the_current_iteration():
    # Line 19 writes H after the line 15 evaluation and the line 16-18 accept
    # test; line 4 of the NEXT iteration is the first read. A sibling recorded
    # mid-iteration must not redefine this iteration's tiers -- otherwise one
    # sibling flipping an example from {0} to {0,1} moves it Hard -> Mixed and
    # can collapse D_eval onto that single example.
    tracker = RejectedHistoryTracker(perfect_score=1.0)
    tracker.record([0, 1, 2], [1.0, 0.0, 0.0])
    assert tracker.entries == []

    tracker.reconcile(pool_size=0)
    assert len(tracker.entries) == 1


def test_reconcile_drops_accepted_proposals_to_avoid_double_counting():
    # An accepted proposal is already in prog_candidate_val_subscores with full
    # D_eval coverage; promoting its minibatch entry too would put the same
    # prompt in H twice.
    tracker = RejectedHistoryTracker(perfect_score=1.0)
    tracker.record([0], [1.0])
    tracker.record([1], [0.0])
    # One of the two was accepted (pool grew by 1).
    tracker.reconcile(pool_size=1)
    assert len(tracker.entries) == 1


def test_history_merges_accepted_candidates_and_rejected_proposals():
    tracker = RejectedHistoryTracker(perfect_score=1.0)
    tracker.record([0], [1.0])
    tracker.reconcile(pool_size=0)

    history = build_history(
        accepted_subscores=[{0: 0.0, 1: 1.0}],
        rejected=tracker.entries,
        perfect_score=1.0,
    )
    # id 0: fail under the seed, pass under the rejected proposal -> Mixed.
    assert tier_of(history.local_history(0, 5)) == "M"
    assert tier_of(history.local_history(1, 5)) == "E"


# ----------------------------------------------------------------------
# Section 4.2 / Eq. 8: the mutation pool
# ----------------------------------------------------------------------


def test_mutation_pool_prefers_mixed_fail_over_hard_fail():
    # Table 3: Hard-only sampling scores 30.3 vs 42.9 for uniform random, so
    # B[M,0] must be exhausted before B[H,0] is touched.
    subscores = [{0: 1.0, 1: 0.0, 2: 0.0}, {0: 0.0, 1: 0.0, 2: 0.0}]
    state = make_state(subscores)
    sampler = ApexDynamicSampling(n=1, minibatch_size=1, lookback=5, perfect_score=1.0)

    tasks = sampler.sample_tasks(state, FixedSelector(1), None, loader(3))
    # id 0 is Mixed-Fail (passed under the seed, fails now); 1 and 2 are Hard.
    assert tasks[0].minibatch_ids == [0]


def test_usage_history_marks_at_emit_and_resets_when_exhausted():
    subscores = [{0: 0.0, 1: 0.0, 2: 0.0}]
    state = make_state(subscores)
    sampler = ApexDynamicSampling(n=1, minibatch_size=1, lookback=5, perfect_score=1.0)

    seen = []
    for _ in range(3):
        seen.append(sampler.sample_tasks(state, FixedSelector(0), None, loader(3))[0].minibatch_ids[0])
    # U is marked at emit time, so three draws are distinct without a reset.
    assert sorted(seen) == [0, 1, 2]
    assert sampler.pool_resets == 0

    # Fourth draw exhausts the pool and resets U (Section 4.2).
    sampler.sample_tasks(state, FixedSelector(0), None, loader(3))
    assert sampler.pool_resets == 1


def test_empty_frontier_yields_no_tasks():
    # A parent that fails nothing has no addressable frontier; the proposer
    # treats an empty task list as "no proposal this iteration".
    state = make_state([{0: 1.0, 1: 1.0}])
    sampler = ApexDynamicSampling(n=3, minibatch_size=2, perfect_score=1.0)
    assert sampler.sample_tasks(state, FixedSelector(0), None, loader(2)) == []


def test_pool_falls_through_to_the_none_buckets():
    # A parent selected in an iteration whose D_eval never covered its failures
    # has empty B[M,0] and B[H,0]. Those failures sit in B[M,None] / B[H,None]
    # instead, carrying the same "the lineage fails this" evidence, so the pool
    # must continue into them -- otherwise no task is produced and the engine
    # spins without consuming budget (no stock sampling strategy can return an
    # empty list, so that path is otherwise unreachable).
    seed = {0: 0.0, 1: 0.0, 2: 0.0}
    child = {0: 0.0}  # this candidate was only ever scored on id 0
    state = make_state([seed, child])
    sampler = ApexDynamicSampling(n=1, minibatch_size=3, lookback=5, perfect_score=1.0)

    tasks = sampler.sample_tasks(state, FixedSelector(1), None, loader(3))

    assert tasks, "sampler must not return an empty task list here"
    # id 0 is B[H,0] (scored by P_curr, failing); 1 and 2 are B[H,None].
    assert tasks[0].minibatch_ids[0] == 0
    assert set(tasks[0].minibatch_ids) == {0, 1, 2}


def test_sampling_raises_on_empty_trainset():
    state = make_state([{}])
    sampler = ApexDynamicSampling(n=1, minibatch_size=1)
    with pytest.raises(ValueError, match="empty trainset"):
        sampler.sample_tasks(state, FixedSelector(0), None, ListDataLoader([]))


# ----------------------------------------------------------------------
# Section 4.3 / Algorithm 1 lines 8-14: rank-sensitive evaluation
# ----------------------------------------------------------------------


def _mixed_tier_state(i=0):
    """A state whose Mixed tier is non-empty, so the real L8-L14 path runs.

    Seed passes 0-4 and fails 5-9; candidate 1 inverts 0,1 and 5,6. Those four
    ids therefore have mixed histories, while 2-4 stay Easy and 7-9 stay Hard.
    """
    seed = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
    cand1 = {0: 0.0, 1: 0.0, 5: 1.0, 6: 1.0}
    return make_state([seed, cand1], i=i)


def test_eval_batch_applies_eq10_budget_split():
    policy = ApexRankSensitivePolicy(n_eval=4, alpha_0=0.2, lookback=5, perfect_score=1.0)
    batch = policy.get_eval_batch(loader(10), _mixed_tier_state())

    # rho_mix and rho_all are both 0.5 here, so ratio = min(0.2, 0.5, 0.5) = 0.2
    # and k_pos = floor(0.2 * 4) = 0 -- the whole budget goes to the negative
    # set. B[M,0] = [5, 6] is drawn first, then the B[H,0] fallthrough fills the
    # rest of N rather than leaving the batch short.
    assert batch == [5, 6, 7, 8]


def test_eval_batch_is_memoized_per_iteration():
    # Algorithm 1 line 15 scores P_new and P_curr on the SAME D_eval, so every
    # candidate of one iteration must receive one batch.
    policy = ApexRankSensitivePolicy(n_eval=4, lookback=5, perfect_score=1.0)
    state = _mixed_tier_state()
    first = policy.get_eval_batch(loader(10), state)
    alpha_before = policy.alpha
    assert policy.get_eval_batch(loader(10), state) == first
    assert policy.alpha == alpha_before


def test_p_curr_advances_only_on_a_win_on_the_shared_d_eval():
    # Algorithm 1 lines 16-18: the comparison is *on D_eval*, and alpha moves
    # only inside the `if`. A challenger scored on ids disjoint from D_eval
    # cannot be judged and must not displace P_curr.
    policy = ApexRankSensitivePolicy(n_eval=4, alpha_0=0.2, beta=0.03, lookback=5, perfect_score=1.0)
    d_eval = policy.get_eval_batch(loader(10), _mixed_tier_state(i=0))
    assert d_eval == [5, 6, 7, 8]
    assert policy.current_best_idx == 0
    assert policy.alpha == pytest.approx(0.2)

    # Challenger covers only {0, 1}: no overlap with D_eval = [5, 6] -> skipped.
    unjudgeable = _mixed_tier_state(i=1)
    unjudgeable.program_candidates.append({"sys": "c2"})
    unjudgeable.prog_candidate_val_subscores.append({0: 1.0, 1: 1.0})
    unjudgeable.parent_program_for_candidate.append([1])
    policy.get_eval_batch(loader(10), unjudgeable)
    assert policy.current_best_idx == 0
    assert policy.alpha == pytest.approx(0.2)

    # Next iteration adds a further candidate, this one scored on D_eval and
    # strictly better there -> pointer moves, alpha anneals once. The pool only
    # ever grows, so this appends on top of the previous candidate.
    winner = _mixed_tier_state(i=2)
    for scores in ({0: 1.0, 1: 1.0}, {5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0}):
        winner.program_candidates.append({"sys": "x"})
        winner.prog_candidate_val_subscores.append(scores)
        winner.parent_program_for_candidate.append([1])
    policy.get_eval_batch(loader(10), winner)
    assert policy.current_best_idx == 3
    assert policy.alpha == pytest.approx(0.23)


def test_p_curr_does_not_move_on_a_loss():
    policy = ApexRankSensitivePolicy(n_eval=4, alpha_0=0.2, beta=0.03, lookback=5, perfect_score=1.0)
    assert policy.get_eval_batch(loader(10), _mixed_tier_state(i=0)) == [5, 6, 7, 8]

    loser = _mixed_tier_state(i=1)
    loser.program_candidates.append({"sys": "c2"})
    # Seed scores 0.0 on both ids; a tie is not "better than" (strict >).
    loser.prog_candidate_val_subscores.append({5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0})
    loser.parent_program_for_candidate.append([1])
    policy.get_eval_batch(loader(10), loser)
    assert policy.current_best_idx == 0
    assert policy.alpha == pytest.approx(0.2)


def test_alpha_anneals_at_most_once_per_iteration():
    # get_best_program is called several times per iteration (engine logging,
    # parent selection, on_optimization_end); the line 16 step must be idempotent.
    policy = ApexRankSensitivePolicy(n_eval=4, alpha_0=0.2, beta=0.03, lookback=5, perfect_score=1.0)
    policy.get_eval_batch(loader(10), _mixed_tier_state(i=0))

    winner = _mixed_tier_state(i=1)
    winner.program_candidates.append({"sys": "c2"})
    winner.prog_candidate_val_subscores.append({5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0})
    winner.parent_program_for_candidate.append([1])
    policy.get_eval_batch(loader(10), winner)
    alpha_after_first = policy.alpha
    for _ in range(5):
        policy.get_best_program(winner)
    assert policy.alpha == pytest.approx(alpha_after_first)
    assert policy.current_best_idx == 2


def test_n_way_argmax_picks_the_best_sibling():
    # With n > 1 the two-way test of line 16 becomes an (n+1)-way argmax over
    # {P_curr} union {P_new^(1..n)} on the shared D_eval.
    policy = ApexRankSensitivePolicy(n_eval=4, alpha_0=0.2, beta=0.03, lookback=5, perfect_score=1.0)
    assert policy.get_eval_batch(loader(10), _mixed_tier_state(i=0)) == [5, 6, 7, 8]

    siblings = _mixed_tier_state(i=1)
    for scores in (
        {5: 1.0, 6: 0.0, 7: 0.0, 8: 0.0},
        {5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0},
        {5: 0.0, 6: 1.0, 7: 0.0, 8: 0.0},
    ):
        siblings.program_candidates.append({"sys": "x"})
        siblings.prog_candidate_val_subscores.append(scores)
        siblings.parent_program_for_candidate.append([1])
    policy.get_eval_batch(loader(10), siblings)
    # Candidate 3 (index 2 + 1) scores 1.0 on both -> the argmax.
    assert policy.current_best_idx == 3
    # One pointer move -> one anneal, regardless of how many siblings won.
    assert policy.alpha == pytest.approx(0.23)


def test_finalize_settles_the_last_iterations_comparison():
    # _resolve_p_curr runs at the START of an iteration, so candidates added in
    # the final iteration are never compared -- the run ends before the next
    # get_eval_batch. finalize() triggers that last comparison; without it a
    # candidate that legitimately beat P_curr on the shared D_eval is discarded.
    policy = ApexRankSensitivePolicy(n_eval=4, alpha_0=0.2, beta=0.03, lookback=5, perfect_score=1.0)
    d_eval = policy.get_eval_batch(loader(10), _mixed_tier_state(i=0))
    assert policy.current_best_idx == 0

    # A final-iteration candidate that wins on D_eval, with no further
    # get_eval_batch call after it.
    final = _mixed_tier_state(i=0)
    final.program_candidates.append({"sys": "late"})
    final.prog_candidate_val_subscores.append(dict.fromkeys(d_eval, 1.0))
    final.parent_program_for_candidate.append([0])

    # Without finalize the win is invisible (same state.i -> memo short-circuit).
    assert policy.get_best_program(final) == 0

    policy.finalize(final)
    assert policy.current_best_idx == 2
    assert policy.alpha == pytest.approx(0.23)

    # Idempotent: no new contenders on a second call.
    policy.finalize(final)
    assert policy.current_best_idx == 2
    assert policy.alpha == pytest.approx(0.23)


def test_p_curr_starts_at_the_seed():
    policy = ApexRankSensitivePolicy(n_eval=4, lookback=5, perfect_score=1.0)
    assert policy.current_best_idx == 0  # Algorithm 1 line 1: P_curr <- P_0


def test_full_eval_when_budget_exceeds_dataset():
    policy = ApexRankSensitivePolicy(n_eval=100, lookback=5, perfect_score=1.0)
    assert policy.get_eval_batch(loader(10), _mixed_tier_state()) == list(range(10))


def test_first_iteration_samples_a_subset_not_all_of_d():
    # The first iteration has no Mixed tier by construction: H holds one entry
    # (the seed's full evaluation of D, line 2), so every R_i has length 1 and
    # Eq. 7 can only yield Easy or Hard. All five buckets lines 8/12/13 draw from
    # are empty, and without the B[H,0] / B[E,1] fallthrough D_eval comes out
    # empty and degenerates to a full pass over D -- paid once per proposal.
    state = make_state([dict.fromkeys(range(10), 0.0)])
    policy = ApexRankSensitivePolicy(n_eval=4, lookback=5, perfect_score=1.0)

    batch = policy.get_eval_batch(loader(10), state)

    assert len(batch) == 4, "must respect N rather than falling back to |D|"
    assert policy.degenerate_fallbacks == 0
    # rho_mix is 0 (no Mixed tier), so Eq. 10 gives k_pos = 0 and the whole
    # budget goes to the negative set -- here B[H,0], the examples the seed fails.
    assert all(data_id in range(10) for data_id in batch)


def test_first_iteration_anchors_on_easy_passes_when_budget_allows():
    # Mirror of the above with a seed that passes everything: the negative
    # buckets are empty, so D_pos falls through to B[E,1].
    state = make_state([dict.fromkeys(range(10), 1.0)])
    policy = ApexRankSensitivePolicy(n_eval=4, alpha_0=1.0, lookback=5, perfect_score=1.0)

    batch = policy.get_eval_batch(loader(10), state)

    assert len(batch) == 4
    assert policy.degenerate_fallbacks == 0


def test_fallback_still_covers_a_truly_empty_stratification():
    # The full-set fallback stays as a backstop for the case no bucket can fill:
    # ids with no history at all have no tier and belong to no bucket.
    state = make_state([{}])
    policy = ApexRankSensitivePolicy(n_eval=4, lookback=5, perfect_score=1.0)

    batch = policy.get_eval_batch(loader(10), state)

    assert batch == list(range(10))
    assert policy.degenerate_fallbacks == 1


def test_seed_eval_batch_hook_is_absent():
    # Algorithm 1 line 2 evaluates the seed on all of D; every tier depends on
    # it, so the policy must not narrow the seed evaluation.
    policy = ApexRankSensitivePolicy(n_eval=4)
    assert not hasattr(policy, "get_seed_eval_batch")


def test_policy_requests_parent_reevaluation():
    # The flag the engine probes to satisfy Algorithm 1 line 15.
    assert ApexRankSensitivePolicy(n_eval=4).requires_parent_reeval is True


@pytest.mark.parametrize("bad", [0, -1])
def test_policy_rejects_invalid_budget(bad):
    with pytest.raises(ValueError, match="n_eval"):
        ApexRankSensitivePolicy(n_eval=bad)


# ----------------------------------------------------------------------
# Algorithm 1 lines 16-17: hill-climbing selection
# ----------------------------------------------------------------------


def test_best_program_compares_on_shared_coverage_only():
    # Averaging each candidate over its own coverage would favor the seed, whose
    # full pass over D includes easy ids the challenger never saw.
    seed = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0}
    # Strictly better on the two ids they share, but a lower own-coverage mean.
    challenger = {3: 1.0, 4: 1.0}
    state = make_state([seed, challenger])
    policy = ApexRankSensitivePolicy(n_eval=4, lookback=5, perfect_score=1.0)

    assert policy.get_best_program(state) == 1
    assert ApexCurrentBestSelector(policy).select_candidate_idx(state) == 1


def test_best_program_keeps_incumbent_on_a_tie():
    seed = {0: 1.0, 1: 0.0}
    tied = {0: 1.0, 1: 0.0}
    policy = ApexRankSensitivePolicy(n_eval=4, lookback=5, perfect_score=1.0)
    assert policy.get_best_program(make_state([seed, tied])) == 0


def test_best_program_skips_candidates_with_no_overlap():
    seed = {0: 1.0, 1: 1.0}
    disjoint = {8: 1.0, 9: 1.0}
    policy = ApexRankSensitivePolicy(n_eval=4, lookback=5, perfect_score=1.0)
    assert policy.get_best_program(make_state([seed, disjoint])) == 0


# ----------------------------------------------------------------------
# Appendix C: two-step Critique -> Mutate
# ----------------------------------------------------------------------


class ScriptedLM:
    """Replies in the paper's output formats; records each batched round."""

    def __init__(self, critique_ok=True, mutate_ok=True):
        self.rounds: list[list[str]] = []
        self.critique_ok = critique_ok
        self.mutate_ok = mutate_ok

    def batch_complete(self, messages_list):
        prompts = [m[0]["content"] for m in messages_list]
        self.rounds.append(prompts)
        out = []
        for prompt in prompts:
            if "Adaptive Prompt Editor" in prompt:
                out.append("<new_instruction>revised</new_instruction>" if self.mutate_ok else "no tags")
            elif self.critique_ok:
                out.append(
                    "<actionable_feedback>\n"
                    '**Locator:** "base"\n'
                    "**Diagnosis:** Type 1 Ambiguity.\n"
                    "**Instruction:** Be specific.\n"
                    "</actionable_feedback>"
                )
            else:
                out.append("no tags")
        return out

    def __call__(self, prompt):
        return self.batch_complete([[{"role": "user", "content": prompt}]])[0]


ERROR_RECORD = {
    "Inputs": "2+2?",
    "Generated Outputs": "5",
    "Feedback": "wrong",
    "Score": 0.0,
}


def test_error_case_uses_listing6_layout():
    rendered = render_error_case(ERROR_RECORD)
    assert "### Failure Example (Score: 0.0)" in rendered
    assert "<input>\n2+2?\n</input>" in rendered
    assert "<actual_output>\n5\n</actual_output>" in rendered
    assert "<critique>\nwrong\n</critique>" in rendered


def test_error_case_falls_back_for_unknown_schemas():
    rendered = render_error_case({"custom": "value"})
    assert "custom: value" in rendered


def test_always_acceptance_defers_to_the_deval_comparison():
    # Algorithm 1 scores P_new once, on D_eval (line 15); the error batch E of
    # line 5 only feeds Critique (line 6). GEPA's default gate would instead
    # judge the child on E first -- and E is by construction examples P_curr
    # already fails, so the parent sum is ~0 and a child that also fails them is
    # discarded before D_eval ever sees it.
    still_failing = SimpleNamespace(
        subsample_scores_before=[0.0, 0.0, 0.0],
        subsample_scores_after=[0.0, 0.0, 0.0],
    )
    assert StrictImprovementAcceptance().should_accept(still_failing, None) is False
    assert AlwaysAcceptance().should_accept(still_failing, None) is True

    # A candidate that does improve on the batch is passed through by both, so
    # the change only ever widens what reaches the selection stage.
    improved = SimpleNamespace(
        subsample_scores_before=[0.0, 0.0, 0.0],
        subsample_scores_after=[1.0, 0.0, 0.0],
    )
    assert StrictImprovementAcceptance().should_accept(improved, None) is True
    assert AlwaysAcceptance().should_accept(improved, None) is True


def test_error_case_reads_execution_feedback():
    # The tool_loop adapter names its diagnosis field ``execution_feedback``.
    # While that key was missing from _FEEDBACK_KEYS the critique slot rendered
    # as "N/A" for every case, and since ``input`` is boilerplate shared by all
    # AlfWorld tasks and ``output`` is empty on a timeout, different minibatches
    # produced byte-identical critique prompts -- collapsing a whole iteration
    # of parallel proposals onto one candidate.
    rendered = render_error_case(
        {
            "score": 0.0,
            "input": "shared boilerplate",
            "output": "",
            "execution_feedback": "hit the turn cap; actions: go to desk 1",
        }
    )
    assert "hit the turn cap" in rendered
    assert "N/A\n</critique>" not in rendered


def test_error_case_appends_unconsumed_fields():
    # Fields no slot consumed carry the per-case signal; they must survive.
    rendered = render_error_case(
        {
            "Inputs": "q",
            "Generated Outputs": "a",
            "Feedback": "wrong",
            "Score": 0.0,
            "stop_reason": "max_turns",
            "turns": 10,
        }
    )
    assert "Additional diagnostics:" in rendered
    assert "- stop_reason: max_turns" in rendered
    assert "- turns: 10" in rendered


def test_error_case_truncates_bulk_fields():
    # Trajectories run 10-25 KB; at m cases per prompt they would bury the
    # diagnosis Listing 4 asks for.
    rendered = render_error_case({"Inputs": "q", "Feedback": "f", "trajectory": "x" * 9000})
    assert "[truncated, 9000 chars total]" in rendered
    assert len(rendered) < 5000


def test_distinct_records_render_distinct_prompts():
    # The property that actually matters: two failures that differ only in their
    # execution feedback must not produce the same critique input.
    base = {"score": 0.0, "input": "same", "output": ""}
    a = render_error_case({**base, "execution_feedback": "failed at drawer 1"})
    b = render_error_case({**base, "execution_feedback": "failed at shelf 3"})
    assert a != b


def test_templates_preserve_the_paper_structure():
    for marker in (
        "Type 1: Weak Decision Boundaries",
        "Type 2: Missing Process Instructions",
        "**Locator:**",
        "<actionable_feedback>",
    ):
        assert marker in CRITIQUE_PROMPT_TEMPLATE
    for marker in (
        "Variable Lockdown (Strict)",
        "Logic Lock (Strict)",
        "Contextual Integration (Flexible)",
        "<new_instruction>",
    ):
        assert marker in MUTATION_PROMPT_TEMPLATE


def test_reflection_issues_exactly_two_batched_rounds():
    # n parallel proposals must cost 2 round-trips, not 2n.
    lm = ScriptedLM()
    strategy = ApexTwoStepReflection(lm)
    jobs = [({"sys": f"p{k}"}, {"sys": [dict(ERROR_RECORD)]}, ["sys"]) for k in range(3)]

    results = strategy.reflect_many(jobs)

    assert len(results) == 3
    assert len(lm.rounds) == 2
    assert len(lm.rounds[0]) == 3
    assert len(lm.rounds[1]) == 3
    # The critique's directive reaches the mutation prompt (lines 6 -> 7).
    assert "Type 1 Ambiguity" in lm.rounds[1][0]
    for proposal, next_lm in results:
        assert proposal.new_texts["sys"] == "revised"
        assert next_lm is strategy


def test_unparseable_critique_skips_the_mutation_round():
    lm = ScriptedLM(critique_ok=False)
    strategy = ApexTwoStepReflection(lm)

    results = strategy.reflect_many([({"sys": "p"}, {"sys": [dict(ERROR_RECORD)]}, ["sys"])])

    assert results[0][0].new_texts == {}
    assert len(lm.rounds) == 1


def test_unparseable_mutation_leaves_component_unchanged():
    lm = ScriptedLM(mutate_ok=False)
    strategy = ApexTwoStepReflection(lm)
    results = strategy.reflect_many([({"sys": "p"}, {"sys": [dict(ERROR_RECORD)]}, ["sys"])])
    assert results[0][0].new_texts == {}


def test_reflection_validates_custom_templates():
    with pytest.raises(ValueError, match="Missing placeholder"):
        ApexTwoStepReflection(ScriptedLM(), critique_prompt_template="nothing")
