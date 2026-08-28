# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Rank-sensitive evaluation policy (APEX Section 4.3).

Faithful implementation of the selection half of *APEX: Automated Prompt
Engineering eXpert with Dynamic Data Selection* (Wang et al.,
arXiv:2606.11459v1), Algorithm 1 lines 8-14 and Section 4.3.

Why subset evaluation
---------------------
The selection stage is "the primary computational overhead of the optimization
loop, often accounting for more than 90% of total costs" (Section 3.2). That
creates a trade-off: full evaluation ranks precisely but exhausts the budget in
a few generations, while random subsampling introduces noise that causes rank
inversion, retaining inferior prompts.

The paper's insight is that absolute performance estimates are largely
redundant -- what selection needs is the *relative rank* of candidates. Because
the population consists of parents, children, and siblings sharing structure,
their performance is identical on most of ``D``. The bottleneck is isolating
discriminative data (Eq. 5)::

    D_disc = { x_i | exists P_a, P_b : s(P_a, x_i) != s(P_b, x_i) }

Evaluating outside ``D_disc`` yields zero selection information.

Construction of D_eval
----------------------
Algorithm 1 lines 8-14 fills a per-iteration budget ``N``:

1. ``D_req = B[M, None]`` -- volatile unknowns: historically Mixed but skipped
   in the previous pass. "These provide the highest information gain and form
   the required baseline."
2. ``R = N - |D_req|`` -- the remaining budget.
3. Stratified sampling balances stability against error correction, governed by
   an anchor ratio clamped against the mixed-tier and global pass rates
   (Eq. 10)::

       k_pos = floor(min(alpha_t, rho_mix, rho_all) * R),  k_neg = R - k_pos

4. ``D_pos`` prioritizes ``B[M, 1]`` (catch regressions), then ``B[E, None]``.
5. ``D_neg`` prioritizes ``B[M, 0]`` (confirm fixes), then ``B[H, None]``.

Both lists carry one extra bucket at the end (``B[E, 1]`` and ``B[H, 0]``) that
only ever supplies ids on the first iteration, where the Mixed tier is empty by
construction and the paper's five source buckets are all empty. See
``_build_batch`` for why that boundary needs covering and what it costs when it
is not.

The anchor ratio anneals upward on every successful update (Eq. 9)::

    alpha_{t+1} = alpha_t + beta * I(P_new > P_curr)

"As the prompt improves, alpha increases, effectively 'locking in' mastered
logic by dedicating more evaluation budget to preventing regressions."

Incremental evaluation
----------------------
Section 4.3 ends with: "If an outcome s(P_curr, x_i) for a sampled point
already exists in the history H, we retrieve it directly from memory, executing
full inference only for the new candidate." GEPA already does exactly this when
``cache_evaluation=True`` -- its metric-call counter charges only cache misses
-- so this policy adds no caching of its own.

One D_eval per iteration
------------------------
Algorithm 1 line 15 evaluates ``P_new`` *and* ``P_curr`` on the same
``D_eval``. GEPA calls ``get_eval_batch`` once per candidate being evaluated, so
without memoization the candidates of one iteration would each receive a
different subset and their scores would not be comparable. The batch is
therefore computed once per iteration and reused, keyed on ``state.i`` (stable
within an iteration).

Hill-climbing, not ranking
--------------------------
``P_curr`` is a single variable (line 1), advanced only when line 16's
comparison *on D_eval* succeeds (line 17), and returned at line 20. There is no
"pick the best of the pool" step anywhere in Algorithm 1; ``Top-k`` (Eq. 3)
describes the *prior* methods of Section 3.1 that APEX improves on.

This matters because per-candidate averages are not comparable here. Under
subset evaluation candidates cover different and deliberately biased slices of
``D`` -- Eq. 10 favors B[M,1] / B[M,0] and never samples B[H,0], so a subset is
systematically easier than ``D`` as a whole. Ranking those averages lets a
candidate scored on ``N`` easy ids beat one scored on all of ``D``. Comparing on
one shared ``D_eval`` instead makes the bias apply to both sides, where it
cancels.

Because GEPA's per-instance Pareto frontier is built from exactly those
non-comparable absolute scores, it carries no meaning under this policy: sparse
and mutually disjoint coverage fragments it until nearly every candidate sits on
some part of the front. APEX does not consult it -- parent selection goes
through ``get_best_program`` (i.e. ``P_curr``) -- and neither should anyone
reading a run's output.

Requirements and deviations
---------------------------
* ``D`` must be a single dataset (``valset=None``), as for the mutation half.
* ``N`` should stay in the paper's *ratio* to ``|D|`` (the paper uses N=100 with
  |D| between 500 and 700, i.e. 15-20%). Setting ``N >= |D|`` degrades this
  policy to full evaluation and disables Section 4.3 entirely.
* ``get_seed_eval_batch`` is deliberately **not** implemented, so the seed keeps
  its full evaluation of ``D`` (Algorithm 1 line 2). Every tier depends on it.
* ``alpha`` anneals exactly when ``P_curr`` moves, matching line 18's position
  inside the ``if`` of line 16.
* ``|D_req| > N`` is undefined in the paper (``R`` would go negative); ``D_req``
  is truncated to ``N`` and ``R`` clamped at 0.
* The first iteration has no Mixed tier at all (``H`` holds one entry, so every
  ``R_i`` has length 1 and Eq. 7 yields only Easy or Hard). The paper leaves this
  undefined; ``D_neg``/``D_pos`` fall through to ``B[H, 0]``/``B[E, 1]`` for that
  one iteration rather than degenerating to a full pass over ``D``.
* With ``n > 1`` proposals per iteration, line 16's two-way test becomes an
  ``(n+1)``-way argmax over ``{P_curr} union {P_new^(1..n)}`` on the shared
  ``D_eval``. The paper defines no sibling semantics; this keeps one subset for
  all contenders and reduces to line 16 exactly at ``n == 1``.
* Requires the engine to re-score ``P_curr`` on each iteration's ``D_eval``
  (line 15), which ``requires_parent_reeval`` requests.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import cast

from gepa.core.data_loader import DataId, DataInst, DataLoader
from gepa.core.state import GEPAState, ProgramIdx
from gepa.strategies.apex_stratification import (
    ExampleId,
    RejectedHistoryTracker,
    Stratification,
    build_history,
    stratify,
)
from gepa.strategies.eval_policy import EvaluationPolicy


class ApexRankSensitivePolicy(EvaluationPolicy[DataId, DataInst]):
    """Select ``N`` rank-sensitive validation ids per iteration (Section 4.3).

    Args:
        n_eval: Evaluation budget per iteration (the paper's ``N``) (the paper's ``T`` in Section 5.1,
            set to 100 against ``|D|`` of 500-700). Keep the ratio, not the
            absolute value: ``N >= |D|`` disables the whole mechanism.
        alpha_0: Initial anchor ratio (0.2 in the paper).
        beta: Anneal step applied on a successful update (0.03 in the paper).
        lookback: Lineage lookback window ``k`` (5 in the paper). Should match
            the sampler's.
        perfect_score: Pass threshold for binarizing scores (Section 3.1).
        history: Shared tracker for the rejected-proposal part of ``H``. The
            paper has one ``H`` serving both stages, so pass the *same* instance
            given to :class:`~gepa.strategies.apex_sampling.ApexDynamicSampling`.

    The policy is stateful: it tracks the annealed ``alpha``, the per-iteration
    memoized ``D_eval``, and the rejected-proposal part of ``H``. Construct one
    per run.
    """

    # Signals to the engine that Algorithm 1 line 15 needs the parent
    # re-evaluated on the same ``D_eval`` as its children. Probed via getattr,
    # so policies without it keep GEPA's default behavior.
    requires_parent_reeval = True

    def __init__(
        self,
        n_eval: int = 100,
        alpha_0: float = 0.2,
        beta: float = 0.03,
        lookback: int = 5,
        perfect_score: float = 1.0,
        history: RejectedHistoryTracker | None = None,
    ):
        if n_eval < 1:
            raise ValueError(f"n_eval must be >= 1, got {n_eval}")
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")

        self.n_eval = n_eval
        self.alpha = alpha_0
        self.beta = beta
        self.lookback = lookback
        self.perfect_score = perfect_score

        # Memoized D_eval for the iteration in flight, so every candidate of one
        # iteration is scored on the same subset (Algorithm 1 line 15).
        self._memo_iteration: int | None = None
        self._memo_batch: list[DataId] = []

        # Times the degenerate all-B[H,0] boundary forced a full-set fallback
        # (see get_eval_batch). Exposed so a run can report how often it left
        # the paper's defined regime.
        self.degenerate_fallbacks = 0

        # P_curr (Algorithm 1 line 1: ``P_curr <- P_0``). A single index, not a
        # ranking over the pool: the paper has no "pick the best candidate"
        # step. It advances only through the line 16 comparison, in
        # ``_advance_p_curr``.
        self._p_curr_idx: int = 0
        # Iteration whose line 16 comparison has already been applied, so
        # repeated ``get_best_program`` calls within one iteration neither
        # re-compare nor double-anneal alpha.
        self._p_curr_iteration: int | None = None
        # Candidate-pool size at the last comparison: candidates added since
        # then are this iteration's P_new set.
        self._pool_size_at_last_compare: int = 1

        # Shared rejected-proposal part of H (Algorithm 1 line 19). Populated by
        # the sampler's observe_proposals hook when the same tracker instance is
        # passed to both; empty otherwise, in which case H covers accepted
        # candidates only.
        self.history = history if history is not None else RejectedHistoryTracker(perfect_score)

    # ------------------------------------------------------------------
    # Stratification against the current prompt
    # ------------------------------------------------------------------

    def _current_key(self, state: GEPAState) -> int:
        """Identify ``P_curr`` for stratification (Algorithm 2 line 10)."""
        return self.get_best_program(state)

    def _stratify(self, data_ids: Sequence[DataId], state: GEPAState) -> Stratification:
        history = build_history(
            accepted_subscores=state.prog_candidate_val_subscores,
            rejected=self.history.entries,
            perfect_score=self.perfect_score,
        )
        # ``s`` comes straight from H: None means P_curr has no recorded outcome
        # for that id, which is what makes D_req = B[M, None] (line 8) a set of
        # volatile *unknowns* rather than a mix of known strengths.
        return stratify(
            data_ids=cast("Sequence[ExampleId]", data_ids),
            history=history,
            current_key=self._current_key(state),
            lookback=self.lookback,
        )

    # ------------------------------------------------------------------
    # Hill-climbing step (Algorithm 1 lines 15-18)
    # ------------------------------------------------------------------

    def _advance_p_curr(self, state: GEPAState) -> None:
        """Apply the line 16 comparison, updating ``P_curr`` and ``alpha``.

        Algorithm 1 lines 15-18::

            15  Evaluate P_new and P_curr on D_eval
            16  if P_new is better than P_curr on D_eval then
            17      P_curr <- P_new
            18      alpha <- alpha + beta

        The comparison is *on D_eval* -- one subset, shared by ``P_curr`` and
        every ``P_new``. That is what makes it sound despite ``D_eval`` being a
        biased sample of ``D``: the bias applies to both sides and cancels.
        Absolute per-candidate means over differing coverages are never compared,
        because across iterations ``D_eval`` differs and those means are not
        commensurable.

        With ``n > 1`` the paper's two-way test becomes an ``(n+1)``-way argmax
        over ``{P_curr} union {P_new^(1..n)}`` on the shared ``D_eval``. The paper
        defines no semantics for siblings (Algorithm 1 produces one ``P_new`` per
        iteration); this preserves the property that matters -- one subset, all
        contenders -- and reduces to line 16 exactly when ``n == 1``.

        ``alpha`` advances only when the pointer moves, matching line 18's
        position *inside* the ``if``. Contenders with no overlap with ``D_eval``
        cannot be judged and are skipped rather than assumed better.
        """
        pool_size = len(state.program_candidates)
        contenders = list(range(self._pool_size_at_last_compare, pool_size))
        self._pool_size_at_last_compare = pool_size
        if not contenders:
            return

        subs = state.prog_candidate_val_subscores
        # The subset both sides are judged on: this iteration's D_eval when one
        # was built, else the incumbent's own coverage (first iteration, where
        # the seed's full evaluation of D is the shared ground).
        eval_ids = list(self._memo_batch) if self._memo_batch else list(subs[self._p_curr_idx].keys())
        if not eval_ids:
            return

        def score_on(idx: int) -> float | None:
            scores = subs[idx]
            shared = [scores[i] for i in eval_ids if i in scores]
            if not shared:
                return None
            return sum(shared) / len(shared)

        incumbent = score_on(self._p_curr_idx)
        if incumbent is None:
            return

        best_idx, best_score = self._p_curr_idx, incumbent
        for idx in contenders:
            challenger = score_on(idx)
            if challenger is not None and challenger > best_score:
                best_idx, best_score = idx, challenger

        if best_idx != self._p_curr_idx:
            self._p_curr_idx = best_idx
            self.alpha += self.beta

    # ------------------------------------------------------------------
    # EvaluationPolicy interface
    # ------------------------------------------------------------------

    def get_eval_batch(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState,
        target_program_idx: ProgramIdx | None = None,
    ) -> list[DataId]:
        """Build ``D_eval`` for this iteration (Algorithm 1 lines 8-14).

        Memoized on ``state.i`` so all candidates evaluated during one iteration
        share one subset (line 15). ``target_program_idx`` is ignored: the batch
        is a property of the iteration, not of the candidate being scored.
        """
        if self._memo_iteration == state.i and self._memo_batch:
            return list(self._memo_batch)

        # Settle the previous iteration's line 16 comparison before stratifying:
        # the tiers, and D_req in particular, are defined relative to P_curr.
        self._resolve_p_curr(state)

        data_ids = list(loader.all_ids())
        if not data_ids:
            return []

        # N >= |D| means no subset selection is possible; evaluate everything.
        # This is the degenerate regime the policy is meant to avoid, so callers
        # should keep N well under |D| (the paper uses 15-20% of |D|).
        if self.n_eval >= len(data_ids):
            batch = data_ids
        else:
            batch = self._build_batch(data_ids, state)

        # Undefined boundary in the paper: every source bucket of lines 12-13
        # (B[M,1], B[E,None], B[M,0], B[H,None]) can be empty at once, leaving
        # D_eval empty and the candidates unscored. It happens when every
        # example's tier is Hard and P_curr has already been scored on all of
        # them -- typically a seed that fails everything. The paper's own runs
        # start at 23.6-85.8% accuracy, so this never arises there, and B[H, 0]
        # is deliberately never sampled (it carries no rank information: all
        # candidates fail those examples).
        #
        # Falling back to the full set is GEPA's stock FullEvaluationPolicy
        # behavior rather than a new selection rule, and it self-corrects: once
        # any example passes, the Mixed tier becomes non-empty and the normal
        # Algorithm 1 path resumes. Costs |D| metric calls for that iteration.
        if not batch:
            self.degenerate_fallbacks += 1
            batch = data_ids

        self._memo_iteration = state.i
        self._memo_batch = list(batch)
        return list(batch)

    def _build_batch(self, data_ids: list[DataId], state: GEPAState) -> list[DataId]:
        """The tiered construction of ``D_eval`` (Algorithm 1 lines 8-14)."""
        strat = self._stratify(data_ids, state)

        # L8: volatile unknowns form the required baseline. Truncated to N; the
        # paper leaves |D_req| > N undefined (R would go negative).
        d_req = cast("list[DataId]", list(strat.bucket("M", None)))[: self.n_eval]

        # L9: remaining budget.
        remaining = max(0, self.n_eval - len(d_req))
        if remaining == 0:
            return d_req

        # L10-11: clamp the anchor ratio against observed competence, then split.
        # An undefined PassRate is dropped from the min rather than read as 0.
        # rho_mix is undefined on the first iteration (the Mixed tier is empty by
        # construction), and zeroing the ratio on a statistic computed over
        # nothing would hand the entire budget to the negative set -- which, when
        # the seed passes everything, is itself empty, so D_eval would come out
        # empty and degenerate to a full pass over D.
        rhos = [
            rho
            for rho in (
                strat.pass_rate(strat.tier_ids("M")),
                strat.pass_rate(cast("Iterable[ExampleId]", data_ids)),
            )
            if rho is not None
        ]
        ratio = min([self.alpha, *rhos])
        k_pos = math.floor(max(0.0, ratio) * remaining)
        k_pos = min(k_pos, remaining)
        k_neg = remaining - k_pos

        chosen: set[DataId] = set(d_req)

        # L12: positives catch regressions -- B[M,1] first, then B[E,None].
        # B[E,1] is appended as a first-iteration fallback (see below).
        d_pos = self._take(
            k_pos,
            [strat.bucket("M", 1), strat.bucket("E", None), strat.bucket("E", 1)],
            chosen,
        )
        chosen.update(d_pos)

        # L13: negatives confirm fixes -- B[M,0] first, then B[H,None], then
        # B[H,0] as the same first-iteration fallback.
        #
        # Why the two extra buckets: the Mixed tier is empty *by construction* on
        # the first iteration. H then holds exactly one entry (the seed's full
        # evaluation of D, line 2), so every R_i has length 1 and Set(R_i) cannot
        # contain both 0 and 1 -- Eq. 7 can only yield Easy or Hard. All of D
        # lands in B[E,1] and B[H,0], and the five buckets lines 8/12/13 draw
        # from are all empty, so D_eval comes out empty.
        #
        # The paper hits this too (its Algorithm 1 has the same t=1 state) but
        # never defines it. It also never samples B[H,0] or B[E,1], on the
        # grounds that they carry no rank information -- true once a lineage
        # exists, because every candidate agrees there. On the first iteration
        # that reasoning does not apply: nothing has been compared yet, and
        # B[H,0] (85 examples the seed fails) is precisely where a new candidate
        # can show improvement, while B[E,1] anchors against regression. So the
        # ordering is unchanged and these only supply what the earlier buckets
        # could not.
        #
        # Cost of getting this wrong is not marginal: without them the empty
        # batch falls back to all of D, and with n parallel proposals that is
        # paid n+1 times. One observed run spent 510 of 1103 metric calls (46%)
        # on the first iteration alone, which capped the search at 5 iterations
        # and a lineage depth of 4 -- and APEX depends on depth, since Appendix
        # C's critique prompt asks for "the single most impactful" change per
        # step.
        d_neg = self._take(
            k_neg,
            [strat.bucket("M", 0), strat.bucket("H", None), strat.bucket("H", 0)],
            chosen,
        )
        chosen.update(d_neg)

        # L14: the union, in a deterministic order.
        return [*d_req, *d_pos, *d_neg]

    def _take(
        self,
        count: int,
        buckets: Sequence[Sequence[ExampleId]],
        exclude: set[DataId],
    ) -> list[DataId]:
        """Take ``count`` ids from ``buckets`` in priority order, skipping ``exclude``.

        Buckets are consumed in the order given -- the paper's "prioritizing X
        then Y" -- and each bucket is taken in its stable id order. Returns fewer
        than ``count`` ids when the buckets run dry; the paper specifies no
        further fallback, so the batch is simply smaller than ``N``.
        """
        taken: list[DataId] = []
        if count <= 0:
            return taken
        for bucket in buckets:
            for data_id in cast("Sequence[DataId]", bucket):
                if len(taken) >= count:
                    return taken
                if data_id in exclude:
                    continue
                taken.append(data_id)
        return taken

    def _resolve_p_curr(self, state: GEPAState) -> None:
        """Run :meth:`_advance_p_curr` at most once per iteration."""
        if self._p_curr_iteration == state.i:
            return
        self._p_curr_iteration = state.i
        self._advance_p_curr(state)

    def finalize(self, state: GEPAState) -> None:
        """Settle the last iteration's line 16 comparison.

        ``_resolve_p_curr`` runs at the *start* of an iteration, so candidates
        added in the final iteration would otherwise never be compared: the run
        ends before the next ``get_eval_batch``. Their scores exist and their
        ``D_eval`` is still memoized, so the comparison is available -- it just
        needs to be triggered. Without this, a candidate that legitimately beat
        ``P_curr`` on the shared subset in the last iteration is silently
        discarded and the run reports the previous ``P_curr``.

        Idempotent: a second call finds no new contenders and does nothing.
        """
        self._advance_p_curr(state)

    @property
    def current_best_idx(self) -> ProgramIdx:
        """``P_curr`` -- what Algorithm 1 line 20 returns.

        Read this for the run's final answer rather than ``GEPAResult.best_idx``,
        which is ``max(val_aggregate_scores)`` over each candidate's average on
        *its own* coverage. Under subset evaluation those averages come from
        different, deliberately biased subsets (Eq. 10 favors B[M,1] / B[M,0] and
        never samples B[H,0]), so they are not comparable across candidates: one
        scored on 15 easy ids outranks one scored on all of ``D``.
        """
        return self._p_curr_idx

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Return ``P_curr`` (the variable Algorithm 1 line 17 assigns).

        The paper has no "select the best from the pool" step: ``P_curr`` is a
        single variable advanced only by the line 16 comparison. This method is
        GEPA's interface for "which candidate is best", so it reports that
        variable; the advance itself lives in :meth:`_advance_p_curr`.

        The engine calls this from ``_add_evaluated_program`` (for an
        ``is_best_program`` log flag) and at ``on_optimization_end`` -- both
        reporting paths, so resolving lazily here is safe. The call that actually
        drives the search is parent selection at the start of each iteration, by
        which point the previous iteration's candidates are all in the pool and
        scored on its ``D_eval``.
        """
        self._resolve_p_curr(state)
        return self._p_curr_idx

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Average score of a candidate over its own evaluated ids.

        Reported for logging and progress tracking, where an absolute number per
        candidate is what callers expect. Ranking never uses this -- see
        ``get_best_program`` on why cross-candidate comparison needs shared
        coverage.
        """
        return state.get_program_average_val_subset(program_idx)[0]


__all__ = ["ApexRankSensitivePolicy"]
