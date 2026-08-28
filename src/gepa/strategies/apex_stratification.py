# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Dynamic data stratification shared by the APEX strategies.

Faithful implementation of Section 4.1 and Algorithm 2 of *APEX: Automated
Prompt Engineering eXpert with Dynamic Data Selection* (Wang et al.,
arXiv:2606.11459v1).

The paper's premise is that a datapoint's utility is not fixed: as prompts
evolve, examples that were once informative discriminators converge into
consistently Easy cases while others persist as Hard (intractable) noise.
Stratification turns the evaluation history into three semantic tiers so both
the mutation and the selection stage can target the high-leverage frontier.

Binary outcomes
---------------
The paper defines ``s(P, x) in {0, 1}`` as the binary evaluation outcome
"where only a perfect score yields a pass (1) and any partial credit is
treated as a failure (0)" (Section 3.1). We therefore binarize with
``score >= perfect_score`` and deliberately expose no separate threshold
parameter -- the paper has none.

Local history and tiers
-----------------------
For datapoint ``x_i``, ``H_valid^(i)`` is the sub-sequence of historical
prompts that actually evaluated ``x_i`` (``s(P, x_i) != None``). The local
history ``R_i`` is the outcomes of the ``k`` most recent prompts in that
sub-sequence (Eq. 6)::

    R_i = { s(P, x_i) | P in last_k(H_valid^(i)) }

Note this window is *per example*: it is a dynamic slice of the sparse score
matrix, so the optimizer always sees a consistent sample size of recent
behavior rather than outdated signals -- not the last ``k`` global iterations.

Tiers follow Eq. 7::

    Tier(i) = E (Easy)  if Set(R_i) == {1}
              H (Hard)  if Set(R_i) == {0}
              M (Mixed) otherwise

* **Easy**: consistently solved by the lineage; re-evaluating gives minimal signal.
* **Hard**: consistently failed; currently intractable.
* **Mixed**: volatile -- the rank-sensitive frontier for evaluation and the most
  probable targets for improvement.

Granular buckets
----------------
Algorithm 2 intersects each tier with the outcome under the *current* prompt
``s in {1, 0, None}`` (``None`` = not evaluated in the current pass), yielding
nine disjoint buckets ``B[tier, s]``. For example ``B[M, 0]`` is the
historically Mixed instances that are currently failing (the *addressable
frontier* used for mutation), and ``B[M, None]`` is the volatile unknowns that
form the required baseline of the rank-sensitive evaluation set.

Examples with an empty ``R_i`` (never evaluated by any prompt) have no defined
tier and are excluded from every bucket. Under the paper's Algorithm 1 this
never happens, because line 2 evaluates the seed prompt on all of ``D``; GEPA
reproduces that as long as the seed receives a full evaluation of ``D``.

History assembly
----------------
Algorithm 1 line 19 updates ``H`` with the evaluation of *both* ``P_new`` and
``P_curr`` on ``D_eval``, and it does so unconditionally -- after the
acceptance test on lines 16-18. Rejected candidates therefore contribute to
``H`` as well. In GEPA those two sources live in different places:

* accepted candidates -> ``state.prog_candidate_val_subscores``
* rejected proposals  -> only ever scored on their minibatch, so the sampler
  accumulates them via the ``observe_proposals`` hook

:class:`HistoryView` merges both into one ordered sequence of
``(source_key, scores_by_id)`` entries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from gepa.core.data_loader import ComparableHashable

# Concrete id type for the stratification structures. GEPA's ``DataId`` is a
# bound TypeVar meant for generic containers; these helpers only need ids to be
# hashable with a stable order, so we use the bound itself rather than making
# every dataclass generic.
ExampleId: TypeAlias = ComparableHashable

# Semantic tier of a datapoint (Eq. 7).
Tier: TypeAlias = Literal["E", "H", "M"]

# Outcome under the current prompt: 1 = pass, 0 = fail, None = not evaluated.
CurrentOutcome: TypeAlias = Literal[1, 0] | None

# Key of a granular bucket B[tier, s] (Algorithm 2).
BucketKey: TypeAlias = tuple[Tier, CurrentOutcome]

TIERS: tuple[Tier, ...] = ("E", "H", "M")
CURRENT_OUTCOMES: tuple[CurrentOutcome, ...] = (1, 0, None)


def passed(score: float, perfect_score: float) -> bool:
    """Binarize a score per Section 3.1: only a perfect score is a pass.

    "we define s(P, x) in {0,1} as the binary evaluation outcome of prompt P on
    datapoint x, where only a perfect score yields a pass (1) and any partial
    credit is treated as a failure (0)."
    """
    return score >= perfect_score


@dataclass
class HistoryEntry:
    """One evaluated prompt's binary outcomes over a subset of ``D``.

    ``key`` identifies the prompt that produced the outcomes. Accepted
    candidates use their candidate index; rejected proposals use an opaque
    string, since they never enter ``state.program_candidates``.

    ``outcomes`` is sparse and already binarized (Section 3.1): it holds only
    the ids this prompt actually evaluated, mapped to 1 (pass) or 0 (fail). A
    missing id means ``s(P, x_i) = None`` for this prompt, which is what makes
    it invisible to ``H_valid^(i)``.
    """

    key: int | str
    outcomes: Mapping[ExampleId, int]


@dataclass
class HistoryView:
    """Ordered evaluation history ``H`` assembled from both GEPA sources.

    Order is *global* (creation order across the whole run), not restricted to
    the ancestral chain of the current candidate: the training-data selection
    strategy is deliberately independent of the candidate-selection strategy,
    so ``R_i`` slices the run's history as a whole.

    Accepted candidates come first in candidate-index order, then rejected
    proposals in the order they were observed. Within one iteration the paper
    defines no ordering among siblings (Algorithm 1 produces exactly one
    ``P_new`` per iteration), so any total order here is arbitrary; we keep
    accepted-before-rejected because accepted candidates carry full ``D_eval``
    coverage and rejected ones only a minibatch.
    """

    entries: list[HistoryEntry] = field(default_factory=list)

    def local_history(self, data_id: ExampleId, lookback: int) -> list[int]:
        """Return ``R_i``: outcomes of the last ``k`` prompts that evaluated ``data_id``.

        Implements Eq. 6. Walks the history backwards collecting only entries
        that actually scored ``data_id`` (the ``H_valid^(i)`` sub-sequence), then
        restores chronological order.
        """
        if lookback <= 0:
            return []
        outcomes: list[int] = []
        for entry in reversed(self.entries):
            outcome = entry.outcomes.get(data_id)
            if outcome is None:
                continue
            outcomes.append(outcome)
            if len(outcomes) >= lookback:
                break
        outcomes.reverse()
        return outcomes

    def latest_outcome(self, key: int | str, data_id: ExampleId) -> CurrentOutcome:
        """Look up ``s(P_curr, x_i)`` from ``H``, or ``None`` if unevaluated.

        Algorithm 2 line 10 ("Lookup cached status, yields None if uneval").
        Scans backwards so the most recent entry for ``key`` wins; a candidate
        can appear more than once when its coverage grew across iterations.
        """
        for entry in reversed(self.entries):
            if entry.key != key:
                continue
            outcome = entry.outcomes.get(data_id)
            if outcome is not None:
                return 1 if outcome == 1 else 0
        return None


def tier_of(local_history: Sequence[int]) -> Tier | None:
    """Assign a semantic tier from ``R_i`` (Eq. 7).

    Returns ``None`` for an empty ``R_i`` (never evaluated), which has no
    defined tier and belongs to no bucket.
    """
    if not local_history:
        return None
    distinct = set(local_history)
    if distinct == {1}:
        return "E"
    if distinct == {0}:
        return "H"
    return "M"


@dataclass
class Stratification:
    """The nine disjoint buckets ``B[tier, s]`` of Algorithm 2.

    ``buckets`` maps ``(tier, current_outcome)`` to the ids in that bucket, in
    stable ``data_ids`` order. ``tiers`` records each id's tier so callers can
    compute tier-level statistics (e.g. the mixed-tier pass rate needed by
    Eq. 10) without re-deriving them.

    Ids with no defined tier (empty ``R_i``) appear in neither mapping.
    """

    buckets: dict[BucketKey, list[ExampleId]]
    tiers: dict[ExampleId, Tier]
    # Most recent outcome in ``R_i`` per id, cached at construction so
    # ``pass_rate`` needs no second history walk.
    _last_outcomes: dict[ExampleId, int] = field(default_factory=dict)

    def bucket(self, tier: Tier, current: CurrentOutcome) -> list[ExampleId]:
        """Ids in ``B[tier, current]`` (empty list when the bucket is empty)."""
        return self.buckets.get((tier, current), [])

    def tier_ids(self, tier: Tier) -> list[ExampleId]:
        """All ids in a tier, across the three current-outcome values."""
        return [data_id for current in CURRENT_OUTCOMES for data_id in self.bucket(tier, current)]

    def pass_rate(self, ids: Iterable[ExampleId]) -> float | None:
        """Fraction of ``ids`` whose most recent outcome in ``R_i`` is a pass.

        Used for ``rho_mix = PassRate(B_M)`` and ``rho_all = PassRate(D)``
        (Algorithm 1 line 10). Ids with no defined tier are skipped.

        Returns ``None`` when no id contributes an outcome, which is *not* the
        same as a rate of 0.0. ``rho_mix`` hits this on the first iteration,
        where the Mixed tier is empty by construction: reporting 0.0 there would
        clamp Eq. 10's ``min(alpha, rho_mix, rho_all)`` to zero on the strength
        of a statistic computed over nothing, sending the whole budget to the
        negative set even when the run's actual failures are elsewhere. The paper
        does not define ``PassRate`` over an empty bucket; callers should drop an
        undefined term from the ``min`` rather than treat it as zero.
        """
        total = 0
        passes = 0
        for data_id in ids:
            outcome = self._last_outcomes.get(data_id)
            if outcome is None:
                continue
            total += 1
            passes += outcome
        if total == 0:
            return None
        return passes / total


def stratify(
    data_ids: Sequence[ExampleId],
    history: HistoryView,
    current_key: int | str,
    lookback: int,
) -> Stratification:
    """Partition ``data_ids`` into the nine buckets ``B[tier, s]`` (Algorithm 2).

    ``current_key`` identifies ``P_curr`` in ``history``; its outcome there
    supplies the ``s`` coordinate, read straight from ``H`` via
    :meth:`HistoryView.latest_outcome`. Iterates ``data_ids`` in order so every
    bucket is deterministically ordered.

    ``s = None`` means ``P_curr`` has no recorded outcome for that id -- Section
    4.1's "skipped in the current pass" and Algorithm 2 line 10's "yields None if
    uneval". That is what makes ``D_req = B[M, None]`` (line 8) a set of *volatile
    unknowns*: Mixed-tier ids whose behaviour under ``P_curr`` is not yet known,
    which is why they must be evaluated before anything else.

    An earlier version scoped ``s`` to the previous iteration's ``D_eval``,
    forcing ``None`` for ids outside it even when ``H`` held a score. Once the
    engine began re-scoring ``P_curr`` each iteration (Algorithm 1 line 15), its
    coverage grew past any single ``D_eval`` and that scoping inverted the
    meaning of ``None``: ids ``P_curr`` had already answered -- and answered well
    -- landed in ``D_req``, so the "unbiased baseline" filled up with incumbent
    strengths and no challenger could win line 16.
    """
    buckets: dict[BucketKey, list[ExampleId]] = {}
    tiers: dict[ExampleId, Tier] = {}
    last_outcomes: dict[ExampleId, int] = {}

    for data_id in data_ids:
        local = history.local_history(data_id, lookback)
        tier = tier_of(local)
        if tier is None:
            # No valid history: no defined tier, belongs to no bucket.
            continue
        tiers[data_id] = tier
        last_outcomes[data_id] = local[-1]
        current = history.latest_outcome(current_key, data_id)
        buckets.setdefault((tier, current), []).append(data_id)

    return Stratification(buckets=buckets, tiers=tiers, _last_outcomes=last_outcomes)


def build_history(
    accepted_subscores: Sequence[Mapping[ExampleId, float]],
    rejected: Sequence[HistoryEntry],
    perfect_score: float,
) -> HistoryView:
    """Assemble ``H`` from GEPA's two evaluation records (Algorithm 1 line 19).

    ``accepted_subscores`` is ``state.prog_candidate_val_subscores`` -- the
    per-instance scores of every candidate in the pool, indexed by candidate
    idx. ``rejected`` holds the minibatch outcomes of proposals that were
    evaluated but never entered the pool; the paper's line 19 updates ``H``
    unconditionally, after the acceptance test, so those belong in ``H`` too.

    Accepted candidates are emitted first in candidate-index order, then the
    rejected entries in observation order. See :class:`HistoryView` on why any
    intra-iteration order is arbitrary.
    """
    entries: list[HistoryEntry] = [
        HistoryEntry(
            key=candidate_idx,
            outcomes={data_id: (1 if passed(score, perfect_score) else 0) for data_id, score in subscores.items()},
        )
        for candidate_idx, subscores in enumerate(accepted_subscores)
    ]
    entries.extend(rejected)
    return HistoryView(entries=entries)


class RejectedHistoryTracker:
    """Accumulates the part of ``H`` that GEPA does not persist.

    Algorithm 1 line 19 updates ``H`` with the evaluation of ``P_new``
    unconditionally -- after the acceptance test on lines 16-18 -- so rejected
    candidates belong in ``H`` too. GEPA only stores per-instance scores for
    candidates that enter the pool, so rejected proposals must be captured as
    they are evaluated, via the proposer's ``observe_proposals`` hook.

    The paper has exactly one ``H`` shared by the mutation and selection stages.
    A single instance of this tracker should therefore be shared between
    :class:`~gepa.strategies.apex_sampling.ApexDynamicSampling` and
    :class:`~gepa.strategies.apex_eval_policy.ApexRankSensitivePolicy`, so both
    stratify against the same history rather than two divergent views.

    Only the sampler receives ``observe_proposals``; the policy reads the
    tracker. When no sampler is wired up the tracker simply stays empty and
    ``H`` degrades to accepted candidates only.
    """

    def __init__(self, perfect_score: float = 1.0):
        self.perfect_score = perfect_score
        self.entries: list[HistoryEntry] = []
        # Candidate-pool size at the previous reconciliation, used to identify
        # how many of the recorded proposals were subsequently accepted.
        self._pool_size = 0
        # Proposals recorded during the iteration in flight, not yet part of H.
        # Promoted by ``reconcile`` at the start of the next iteration.
        self._pending: list[HistoryEntry] = []
        # Monotonic counter for entry keys, so a promoted entry keeps a stable
        # identity even after earlier entries are dropped.
        self._next_key = 0

    def record(self, ids: Sequence[ExampleId], scores: Sequence[float]) -> None:
        """Record one evaluated proposal's minibatch outcomes as *pending*.

        Pending entries are withheld from :attr:`entries` until :meth:`reconcile`
        promotes them. Algorithm 1 writes ``H`` on line 19, after the line 15
        evaluation and the line 16-18 acceptance test, and the updated ``H`` is
        first read by line 4 of the *next* iteration. Admitting a proposal's
        outcomes into ``H`` immediately would instead let the current iteration's
        own siblings redefine the tiers that lines 8-14 are selecting against --
        a single sibling flipping one example's history from ``{0}`` to
        ``{0, 1}`` moves it from Hard to Mixed and can shrink ``D_eval`` to that
        one example.
        """
        if len(ids) != len(scores):
            return
        self._pending.append(
            HistoryEntry(
                key=f"proposal:{self._next_key}",
                outcomes={
                    data_id: (1 if passed(score, self.perfect_score) else 0)
                    for data_id, score in zip(ids, scores, strict=True)
                },
            )
        )
        self._next_key += 1

    def reconcile(self, pool_size: int) -> None:
        """Promote the previous iteration's pending entries into ``H``.

        Called at the start of an iteration, which is where line 19's write
        becomes visible to line 4. Accepted proposals are dropped rather than
        promoted: they are already in ``state.prog_candidate_val_subscores`` with
        strictly better coverage (their full ``D_eval`` instead of one
        minibatch), so promoting them too would enter the same prompt into ``H``
        twice.

        Identifying them by pool growth is approximate -- it assumes the accepted
        candidates are among the most recently recorded proposals, which holds
        because the engine adds them in the same iteration that recorded them.
        """
        num_accepted = max(0, pool_size - self._pool_size)
        promote = self._pending[: max(0, len(self._pending) - num_accepted)]
        self.entries.extend(promote)
        self._pending = []
        self._pool_size = pool_size


def shared_subset_average(
    subscores: Mapping[ExampleId, float],
    reference_ids: Iterable[ExampleId],
) -> tuple[float, int]:
    """Average ``subscores`` over ``reference_ids`` only, plus the overlap size.

    Under a subset evaluation policy, candidates cover different -- and
    deliberately biased -- slices of ``D``: the seed gets a full evaluation
    while later candidates see only ``N`` ids skewed toward the Mixed/Fail
    frontier. Averaging each candidate over *its own* coverage therefore
    compares scores computed on non-comparable samples and systematically
    favors the seed.

    Algorithm 1 line 15 avoids this by evaluating ``P_new`` and ``P_curr`` on
    the *same* ``D_eval``; restricting both averages to a shared id set is the
    same comparison. Returns ``(-inf, 0)`` when there is no overlap.
    """
    overlap = [subscores[data_id] for data_id in reference_ids if data_id in subscores]
    if not overlap:
        return float("-inf"), 0
    return sum(overlap) / len(overlap), len(overlap)


__all__ = [
    "CURRENT_OUTCOMES",
    "TIERS",
    "BucketKey",
    "CurrentOutcome",
    "HistoryEntry",
    "HistoryView",
    "RejectedHistoryTracker",
    "Stratification",
    "Tier",
    "build_history",
    "passed",
    "shared_subset_average",
    "stratify",
    "tier_of",
]
