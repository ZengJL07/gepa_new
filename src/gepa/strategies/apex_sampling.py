# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Trajectory-guided mutation sampling (APEX Section 4.2).

Faithful implementation of the mutation half of *APEX: Automated Prompt
Engineering eXpert with Dynamic Data Selection* (Wang et al.,
arXiv:2606.11459v1), Algorithm 1 lines 4-5 and Section 4.2.

Why not random error sampling
-----------------------------
Standard methods sample errors uniformly from the entire failure set. The paper
argues this makes mutation "a source of instability rather than monotonic
improvement" for two reasons: optimization follows a hierarchy of "fixability"
(errors in ``E_b`` may only be addressable once ``E_a`` is resolved, so
reversing that order causes high-variance updates and forgetting), and uniform
sampling frequently retrieves "impossible" cases beyond the target model's
capacity, which misleads the meta-optimizer and stalls progress.

The addressable frontier
------------------------
APEX instead builds the error batch primarily from the Mixed-Fail bucket
``B[M, 0]``: "soft failures" where the model demonstrated capacity in the recent
lineage but regressed under the current variation. Correcting these regressions
stabilizes the trajectory. Errors are drawn from ``B[M, 0] union B[H, 0]``
(Eq. 8), Mixed first.

The paper's ablation (Table 3) is the reason the union is ordered rather than
uniform: sampling only the Hard tier scores 30.3 versus 42.9 for uniform random
over all data, because the prompt overfits to narrow edge cases. Mixed data
"provides a vital grounding signal, preventing catastrophic overfitting".

Coverage maximization
---------------------
A usage history ``U`` prevents the optimizer from repeatedly overfitting to the
same recurring failures. Errors come exclusively from unvisited failures
(Eq. 8)::

    e in { x_i | x_i in (B[M,0] union B[H,0]), x_i not in U }

"Once this pool is exhausted, the usage history is reset, guaranteeing broad
exploration of the error surface."

Deviations from the paper, forced by GEPA's architecture
--------------------------------------------------------
* **n > 1.** Algorithm 1 is strictly sequential: one ``P_new`` per iteration.
  This strategy emits ``n`` tasks per iteration to match GEPA's parallel
  proposal mode. The paper defines no semantics for siblings, so ``U`` is
  marked at *emit* time and consulted per draw -- the literal reading of Eq. 8
  applied ``n`` times. Consequence: if the pool empties mid-iteration, ``U``
  resets and a later task may redraw an id an earlier task already used.
* **GEPA evaluates the minibatch.** The paper's mutation stage performs no
  evaluation at all (footnote 1: "generating a mutation often requires fewer
  than three API calls") because the buckets already record which examples
  fail. GEPA needs a traced parent evaluation to build the reflective dataset,
  so it re-evaluates the minibatch. This costs extra metric calls and makes
  ``H`` denser than the paper's.
* **Rejected proposals enter ``H`` from memory.** Algorithm 1 line 19 updates
  ``H`` unconditionally, including for rejected candidates. Those never enter
  ``state.program_candidates``, so they are accumulated here via
  ``observe_proposals`` and are lost on resume (the paper has no resume).

Requirements
------------
``D`` must be a single dataset (pass ``valset=None`` so GEPA reuses the
trainset). The paper's ``D`` serves both mutation and selection; the nine
buckets require tiers and current outcomes over the *same* examples. With a
disjoint trainset the parent has almost no per-example coverage there, so
``B[M, 0]`` would be permanently empty.

The seed must receive a full evaluation of ``D`` (Algorithm 1 line 2,
``EvaluateFull(P_0, D)``), which GEPA does by default. Without it every ``R_i``
is empty, no example has a tier, and the mutation pool starts empty.
"""

from __future__ import annotations

from typing import Any, Generic, cast

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.apex_stratification import (
    HistoryView,
    RejectedHistoryTracker,
    Stratification,
    build_history,
    stratify,
)
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.proposal_sampling import ProposalTask, SamplingStrategy


class ApexDynamicSampling(SamplingStrategy, Generic[DataId, DataInst]):
    """Sample mutation minibatches from the addressable frontier ``B[M,0] union B[H,0]``.

    Emits ``n`` tasks per iteration. Each task selects a parent via
    ``candidate_selector``, stratifies ``D`` against that parent (Algorithm 2),
    and draws ``minibatch_size`` unvisited failures, Mixed-Fail first (Eq. 8).

    Args:
        n: Tasks (parallel proposals) per iteration. The paper uses 1.
        minibatch_size: Errors per mutation batch (the paper's ``m``, 5).
        lookback: Lineage lookback window (the paper's ``k``, 5). Table 4 reports
            50.3 / 52.3 / 50.6 for k = 3 / 5 / 10 -- a wider window reintroduces
            stale signals.
        perfect_score: Pass threshold for binarizing scores. Section 3.1 counts
            only a perfect score as a pass, so this should match the engine's
            ``perfect_score``; it is not a free tuning parameter.
        history: Shared tracker for the rejected-proposal part of ``H``. The
            paper has exactly one ``H`` serving both stages, so pass the *same*
            instance to :class:`~gepa.strategies.apex_eval_policy.ApexRankSensitivePolicy`.
            When omitted a private tracker is created, which is correct only if
            the rank-sensitive policy is not in use.

    Reproducibility: selection is fully deterministic. Buckets are built in
    ``trainset.all_ids()`` order and a draw takes the leading unvisited ids, so
    no RNG is involved -- the paper prescribes a priority order, not a
    distribution.
    """

    def __init__(
        self,
        n: int = 1,
        minibatch_size: int = 5,
        lookback: int = 5,
        perfect_score: float = 1.0,
        history: RejectedHistoryTracker | None = None,
    ):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if minibatch_size < 1:
            raise ValueError(f"minibatch_size must be >= 1, got {minibatch_size}")
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")

        self.n = n
        self.minibatch_size = minibatch_size
        self.lookback = lookback
        self.perfect_score = perfect_score
        self.history = history if history is not None else RejectedHistoryTracker(perfect_score)

        # Usage history U (Eq. 8): ids already drawn for mutation. Marked at
        # emit time so a batch counts as visited even if its proposal is later
        # rejected -- the paper's goal is coverage of the error surface, which
        # is independent of whether a mutation succeeded.
        self.used: set[DataId] = set()
        # Number of times U has been reset (exposed for diagnostics/tests).
        self.pool_resets = 0

    # ------------------------------------------------------------------
    # History (Algorithm 1 line 19)
    # ------------------------------------------------------------------

    def _history(self, state: GEPAState) -> HistoryView:
        """Assemble ``H`` from accepted candidates plus observed rejections."""
        return build_history(
            accepted_subscores=state.prog_candidate_val_subscores,
            rejected=self.history.entries,
            perfect_score=self.perfect_score,
        )

    def observe_proposals(self, proposals: list[Any]) -> None:
        """Record every evaluated proposal's minibatch outcomes into ``H``.

        Called by the proposer with all evaluated proposals for an iteration,
        *before* the engine decides acceptance. Algorithm 1 line 19 updates ``H``
        with the evaluation of ``P_new`` regardless of that decision, so all of
        them belong in ``H``.

        Accepted proposals are reconciled away on the next ``sample_tasks``: they
        also land in ``state.prog_candidate_val_subscores`` with strictly better
        coverage (their full ``D_eval`` rather than just the minibatch), so
        keeping both would enter the same prompt into ``H`` twice.
        """
        for proposal in proposals:
            ids = getattr(proposal, "subsample_indices", None)
            after = getattr(proposal, "subsample_scores_after", None)
            if ids is None or after is None:
                continue
            self.history.record(ids, after)

    # ------------------------------------------------------------------
    # Mutation pool (Eq. 8)
    # ------------------------------------------------------------------

    def _ordered_pool(self, strat: Stratification) -> list[DataId]:
        """The addressable frontier: ``B[M,0]`` then ``B[H,0]`` (Eq. 8).

        Mixed-Fail comes first because those are the regressions the lineage has
        recently solved and can plausibly re-solve. Hard-Fail follows as the
        remainder of the failure set; Table 3 shows drawing from Hard *alone*
        collapses performance (30.3 vs 42.9 for uniform random), so it is a
        fallback, never the primary source.

        The ``s = None`` counterparts are appended last. In the paper they are
        not a separate case: Algorithm 1 line 15 scores ``P_curr`` on the whole
        ``D_eval`` each iteration, so a historically Mixed or Hard example is
        essentially always ``s in {0, 1}`` and Eq. 8's two buckets suffice. Under
        GEPA a parent can be selected in an iteration whose ``D_eval`` never
        covered its failures, which empties both buckets and would make this
        strategy return no task at all -- and the engine then spins without
        consuming budget, since no existing sampling strategy can return an empty
        task list. Examples in ``B[M, None]`` / ``B[H, None]`` carry the same
        "the lineage fails this" evidence as the ``s = 0`` buckets, differing
        only in whether the current pass happened to re-check them, so they are
        the faithful continuation of the same priority order.
        """
        # The stratification helpers key on the concrete ``ExampleId`` bound
        # rather than this class's ``DataId`` TypeVar; the ids are the loader's
        # own, so the cast is a no-op at runtime.
        return cast(
            "list[DataId]",
            [
                *strat.bucket("M", 0),
                *strat.bucket("H", 0),
                *strat.bucket("M", None),
                *strat.bucket("H", None),
            ],
        )

    def _draw(self, pool: list[DataId]) -> list[DataId]:
        """Draw up to ``minibatch_size`` unvisited ids, resetting ``U`` if exhausted.

        Takes the pool's leading ids, which preserves the Mixed-before-Hard
        priority of Eq. 8. Returns fewer than ``minibatch_size`` ids when the
        frontier is smaller than that; an empty result means the parent has no
        known failures at all, and the caller skips the task.
        """
        available = [data_id for data_id in pool if data_id not in self.used]

        # "Once this pool is exhausted, the usage history is reset, guaranteeing
        # broad exploration of the error surface." (Section 4.2)
        if not available and pool:
            self.used.clear()
            self.pool_resets += 1
            available = list(pool)

        return available[: self.minibatch_size]

    # ------------------------------------------------------------------
    # SamplingStrategy interface
    # ------------------------------------------------------------------

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        trainset: DataLoader[DataId, DataInst],
    ) -> list[ProposalTask[DataId, DataInst]]:
        """Emit ``n`` (parent, addressable-frontier minibatch) tasks.

        ``batch_sampler`` is unused: the frontier replaces epoch-shuffled random
        sampling entirely. It stays in the signature to satisfy the protocol.

        A task is skipped when its parent's frontier is empty -- the parent has
        no recorded failure on any example with a defined tier. Returning fewer
        than ``n`` tasks (or none) is safe; the proposer treats an empty list as
        "no proposal this iteration".
        """
        data_ids = list(trainset.all_ids())
        if not data_ids:
            raise ValueError("Cannot sample a minibatch from an empty trainset.")

        # Fold the previous iteration's acceptances into H before stratifying.
        self.history.reconcile(len(state.program_candidates))
        history = self._history(state)

        tasks: list[ProposalTask[DataId, DataInst]] = []
        for _ in range(self.n):
            parent_idx = candidate_selector.select_candidate_idx(state)
            strat = stratify(
                data_ids=data_ids,
                history=history,
                current_key=parent_idx,
                lookback=self.lookback,
            )
            mb_ids = self._draw(self._ordered_pool(strat))

            if not mb_ids:
                continue

            # Mark at emit time: a batch counts as visited even if its proposal
            # is later rejected, and siblings in this iteration see the update.
            self.used.update(mb_ids)

            tasks.append(
                ProposalTask(
                    parent_idx=parent_idx,
                    parent_candidate=state.program_candidates[parent_idx],
                    minibatch_ids=mb_ids,
                    minibatch=trainset.fetch(mb_ids),
                )
            )
        return tasks


__all__ = ["ApexDynamicSampling"]
