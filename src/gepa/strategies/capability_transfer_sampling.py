# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Capability-transfer UCB sampling strategy.

Selects training minibatches by transferring, across the optimization
history, the empirical utility of each training example for *fixing* specific
validation questions.

Intuition (framed statistically, not causally): if training on example ``k``
has historically turned a failing validation question ``j`` into a passing one
for some candidate, that is prior evidence that ``k`` can teach question ``j``.
When we optimize a new parent candidate ``c`` we look at the questions ``c``
currently fails and prefer minibatches whose examples have that prior.

Matrix view (``val_id`` rows x ``train_id`` columns, two counts each):

* ``A[j, k]``: times example ``k`` was in a minibatch whose parent *failed*
  question ``j`` and whose accepted child then *passed* it.
* ``B[j, k]``: same masking (parent failed ``j``), example ``k`` present, but
  the child still did not pass ``j``.

The Beta-smoothed utility is ``mu[j, k] = (A + alpha) / (A + B + alpha + beta)``.
For a parent ``c`` we mask to the rows it fails (``s_cj < tau``) and score each
column by the *mean* utility over those rows.

**Usability gate.** Independently of any validation question, some training
examples are simply poor material to learn from: the parent already solves them
(no headroom — and an all-solved minibatch can never be accepted) or nobody has
ever managed to improve on them (too hard / low-signal). We track, per training
example ``k`` and across *every evaluated proposal* (accepted or rejected), a
column-only usability:

* ``obs[k]``: times ``k`` appeared in an evaluated minibatch.
* ``fix[k]``: of those, times the parent failed ``k`` (``before < tau``) yet the
  child then passed it (``after >= tau``).

    U[k] = (fix[k] + alpha_U) / (obs[k] + alpha_U + beta_U)

Both "parent already solves ``k``" (never counts as a fix) and "parent and child
both fail ``k``" (stuck / too hard) drive ``U[k]`` down, so easy-ceiling and
low-signal examples are implicitly down-weighted without ever hard-excluding
them. Rejected proposals are the strongest "stuck" evidence, so ``U`` is updated
on all evaluated proposals — many more observations than the accept-only ``A``/``B``.

The final per-column score multiplies relevance by usability (a gate: an
example we cannot improve on is down-weighted regardless of which questions it
correlates with) and adds the exploration term *outside* the gate (so cold
columns still get explored rather than being suppressed by a low ``U``):

    value(k)  = U[k]^gamma * mean_{j : parent fails j} mu[j, k]
    Score(k)  = value(k) + explore(k)
    explore(k) = cold_start_bonus                       if n_k == 0
                 exploration_weight * sqrt(log T / n_k)  otherwise

``gamma`` (``usability_weight``) tunes the gate strength; ``gamma == 0`` disables
it (``U^0 == 1``) for ablation. Mean (not sum) over the masked rows keeps the
relevance term in ``[0, 1]`` so ``cold_start_bonus`` and the UCB term stay on the
same scale; since every column shares the identical masked row set within one
scoring call, mean vs. sum is a global constant and does not change the ranking.

Reproducibility: given the same constructor arguments and the same run history,
``sample_tasks`` returns identical minibatches. Ties in ``Score`` are broken by
a seeded RNG.

State ownership: the GEPA engine does not persist which minibatch produced each
accepted candidate, so this strategy is *stateful*. It logs every emitted
``(parent_idx, minibatch_ids)`` edge, increments ``n_k`` at emit time (so a
minibatch counts as sampled even when its proposal is later rejected), and
reconciles newly-accepted candidates into ``A``/``B`` on the next call by
FIFO-per-parent matching. Usability (``obs``/``fix``) is updated separately via
``observe_proposals``, which the proposer calls with *all* evaluated proposals
(including rejected ones) right after minibatch evaluation. Because it is
constructed once and reused across the run, it accumulates history internally
rather than rebuilding from ``state``.
"""

import math
import random
from collections import defaultdict, deque
from typing import Generic

from gepa.core.data_loader import DataId, DataInst, DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.proposal_sampling import ProposalTask, SamplingStrategy


class CapabilityTransferUCBSampling(SamplingStrategy, Generic[DataId, DataInst]):
    """UCB batch sampling driven by cross-candidate capability transfer.

    Emits ``n`` tasks per iteration (``n`` parallel proposals). Each task
    selects a parent via ``candidate_selector`` and then the UCB-optimal
    minibatch of size ``minibatch_size`` for that parent. Within one iteration,
    columns already chosen by earlier tasks are excluded so parallel tasks get
    distinct minibatches (falling back to reuse only if the trainset is too
    small to supply ``n * minibatch_size`` distinct examples).

    Args:
        n: Number of (parent, minibatch) tasks per iteration.
        minibatch_size: Examples per minibatch (fixed batch size).
        tau: Score threshold; a question is "solved" when ``score >= tau``.
        alpha, beta: Beta-prior smoothing for ``mu``.
        exploration_weight: UCB coefficient ``lambda`` on the sqrt term.
        cold_start_bonus: Flat bonus added to never-sampled columns.
        usability_weight: Exponent ``gamma`` on the usability gate ``U[k]``.
            ``0`` disables the gate (``U^0 == 1``); larger values sharpen it.
        alpha_u, beta_u: Beta-prior smoothing for the usability ``U[k]``.
        seed: Seed for the tie-break RNG (reproducibility).
    """

    def __init__(
        self,
        n: int = 1,
        minibatch_size: int = 3,
        tau: float = 0.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        exploration_weight: float = 0.2,
        cold_start_bonus: float = 0.2,
        usability_weight: float = 1.0,
        alpha_u: float = 1.0,
        beta_u: float = 1.0,
        seed: int = 0,
    ):
        assert n >= 1
        assert minibatch_size >= 1
        assert alpha > 0 and beta > 0
        assert alpha_u > 0 and beta_u > 0
        assert usability_weight >= 0
        self.n = n
        self.minibatch_size = minibatch_size
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.exploration_weight = exploration_weight
        self.cold_start_bonus = cold_start_bonus
        self.usability_weight = usability_weight
        self.alpha_u = alpha_u
        self.beta_u = beta_u
        self.rng = random.Random(seed)

        # Capability matrix, sparse. Keys are (val_id, train_id).
        self.A: dict[tuple[DataId, DataId], float] = defaultdict(float)
        self.B: dict[tuple[DataId, DataId], float] = defaultdict(float)
        # Column-only usability counts (updated on every evaluated proposal).
        self.obs: dict[DataId, float] = defaultdict(float)
        self.fix: dict[DataId, float] = defaultdict(float)
        # Times each training example has been sampled (incremented at emit).
        self.n_k: dict[DataId, int] = defaultdict(int)
        # Total task emits (the UCB horizon T).
        self.total_emits = 0

        # Reconciliation bookkeeping. Candidate indices < _reconciled_upto have
        # already been folded into A/B (index 0 is the parent-less seed).
        self._reconciled_upto = 1
        # Emitted-but-unreconciled edges, one FIFO queue per parent idx.
        self._pending: dict[int, deque[tuple[DataId, ...]]] = defaultdict(deque)

    # ------------------------------------------------------------------
    # Reconciliation: fold newly-accepted candidates into A/B.
    # ------------------------------------------------------------------

    def _solved(self, score: float) -> bool:
        return score >= self.tau

    def _reconcile(self, state: GEPAState) -> None:
        """Match new candidates to the minibatches that produced them and update A/B.

        A candidate is reconciled only if it has exactly one parent (reflective
        mutation) and that parent has an unconsumed pending edge — merge
        candidates (two parents) and any candidate we did not emit are skipped.
        Pending edges left unconsumed correspond to rejected proposals; their
        ``n_k`` was already counted at emit, which is intended.
        """
        n_cands = len(state.program_candidates)
        for child_idx in range(self._reconciled_upto, n_cands):
            parents = state.parent_program_for_candidate[child_idx]
            if len(parents) != 1 or parents[0] is None:
                continue
            parent_idx = parents[0]
            queue = self._pending.get(parent_idx)
            if not queue:
                continue
            minibatch_ids = queue.popleft()
            self._update_ab(state, parent_idx, child_idx, minibatch_ids)
        self._reconciled_upto = n_cands

    def _update_ab(
        self,
        state: GEPAState,
        parent_idx: int,
        child_idx: int,
        minibatch_ids: tuple[DataId, ...],
    ) -> None:
        parent_scores = state.prog_candidate_val_subscores[parent_idx]
        child_scores = state.prog_candidate_val_subscores[child_idx]
        for val_id, p_score in parent_scores.items():
            if self._solved(p_score):
                # Row masked: parent already solves this question.
                continue
            if val_id not in child_scores:
                continue
            improved = self._solved(child_scores[val_id])
            for k in minibatch_ids:
                if improved:
                    self.A[(val_id, k)] += 1.0
                else:
                    self.B[(val_id, k)] += 1.0

    # ------------------------------------------------------------------
    # Usability: update from every evaluated proposal (accepted or rejected).
    # ------------------------------------------------------------------

    def observe_proposals(
        self,
        proposals: "list",
    ) -> None:
        """Fold train-batch before/after scores into ``obs``/``fix``.

        Called by the proposer with all evaluated proposals for an iteration,
        before the engine decides acceptance. Each proposal carries
        ``subsample_indices`` aligned with ``subsample_scores_before`` (parent)
        and ``subsample_scores_after`` (child) on the minibatch. A "fix" is a
        train example the parent failed (``before < tau``) and the child then
        passed (``after >= tau``). Rejected proposals are included on purpose:
        they are the strongest evidence that an example is hard to improve on.
        """
        for p in proposals:
            ids = getattr(p, "subsample_indices", None)
            before = getattr(p, "subsample_scores_before", None)
            after = getattr(p, "subsample_scores_after", None)
            if ids is None or before is None or after is None:
                continue
            if len(ids) != len(before) or len(ids) != len(after):
                continue
            for k, b, a in zip(ids, before, after, strict=True):
                self.obs[k] += 1.0
                if (not self._solved(b)) and self._solved(a):
                    self.fix[k] += 1.0

    # ------------------------------------------------------------------
    # Scoring.
    # ------------------------------------------------------------------

    def _mu(self, val_id: DataId, train_id: DataId) -> float:
        a = self.A[(val_id, train_id)]
        b = self.B[(val_id, train_id)]
        return (a + self.alpha) / (a + b + self.alpha + self.beta)

    def _usability(self, train_id: DataId) -> float:
        """Column-only usability U[k] in [0, 1]; gates the relevance term."""
        f = self.fix[train_id]
        o = self.obs[train_id]
        return (f + self.alpha_u) / (o + self.alpha_u + self.beta_u)

    def _explore(self, train_id: DataId) -> float:
        nk = self.n_k[train_id]
        if nk == 0:
            return self.cold_start_bonus
        return self.exploration_weight * math.sqrt(math.log(max(self.total_emits, 1)) / nk)

    def _score_columns(self, state: GEPAState, parent_idx: int, train_ids: list[DataId]) -> dict[DataId, float]:
        parent_scores = state.prog_candidate_val_subscores[parent_idx]
        gap_ids = [j for j, s in parent_scores.items() if not self._solved(s)]
        denom = len(gap_ids)
        scores: dict[DataId, float] = {}
        for k in train_ids:
            if denom > 0:
                relevance = sum(self._mu(j, k) for j in gap_ids) / denom
            else:
                relevance = 0.0
            # Usability gate (multiplicative); gamma == 0 disables it.
            if self.usability_weight > 0:
                gate = self._usability(k) ** self.usability_weight
            else:
                gate = 1.0
            # Exploration is added OUTSIDE the gate so cold columns are still
            # explored rather than suppressed by a low usability.
            scores[k] = gate * relevance + self._explore(k)
        return scores

    def _select_batch(
        self,
        scores: dict[DataId, float],
        train_ids: list[DataId],
        excluded: set[DataId],
    ) -> list[DataId]:
        """Top-``minibatch_size`` columns by score; ties broken by seeded RNG.

        ``excluded`` columns (already used by earlier tasks this iteration) are
        preferred to be skipped, but are reused when too few columns remain.
        """
        # Deterministic per-column tiebreaker draw, in stable train_id order.
        tiebreak = {k: self.rng.random() for k in train_ids}
        available = [k for k in train_ids if k not in excluded]
        if len(available) < self.minibatch_size:
            # Not enough fresh columns; allow reuse of excluded ones (ranked last).
            remainder = [k for k in train_ids if k in excluded]
            ranked = sorted(available, key=lambda k: (-scores[k], tiebreak[k]))
            ranked += sorted(remainder, key=lambda k: (-scores[k], tiebreak[k]))
        else:
            ranked = sorted(available, key=lambda k: (-scores[k], tiebreak[k]))
        return ranked[: self.minibatch_size]

    # ------------------------------------------------------------------
    # SamplingStrategy interface.
    # ------------------------------------------------------------------

    def sample_tasks(
        self,
        state: GEPAState,
        candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        trainset: DataLoader[DataId, DataInst],
    ) -> list[ProposalTask[DataId, DataInst]]:
        # Fold the previous iterations' accepted candidates into A/B first.
        self._reconcile(state)

        train_ids = list(trainset.all_ids())
        if not train_ids:
            raise ValueError("Cannot sample a minibatch from an empty trainset.")

        tasks: list[ProposalTask[DataId, DataInst]] = []
        used_this_iteration: set[DataId] = set()
        for _ in range(self.n):
            parent_idx = candidate_selector.select_candidate_idx(state)
            scores = self._score_columns(state, parent_idx, train_ids)
            mb_ids = self._select_batch(scores, train_ids, used_this_iteration)
            used_this_iteration.update(mb_ids)

            # Record the pull: n_k and horizon advance at emit time, so a batch
            # counts as sampled even if its proposal is later rejected.
            self.total_emits += 1
            for k in mb_ids:
                self.n_k[k] += 1
            self._pending[parent_idx].append(tuple(mb_ids))

            tasks.append(
                ProposalTask(
                    parent_idx=parent_idx,
                    parent_candidate=state.program_candidates[parent_idx],
                    minibatch_ids=mb_ids,
                    minibatch=trainset.fetch(mb_ids),
                )
            )
        return tasks
