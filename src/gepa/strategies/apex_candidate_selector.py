# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Hill-climbing candidate selection for APEX (Algorithm 1 lines 16-17).

Algorithm 1 of *APEX: Automated Prompt Engineering eXpert with Dynamic Data
Selection* (Wang et al., arXiv:2606.11459v1) carries a single ``P_curr``::

    if P_new is better than P_curr on D_eval:
        P_curr <- P_new

That is hill-climbing over one lineage, not GEPA's Pareto population. ``P_curr``
is a single variable, and the ``better than`` of line 16 is qualified: *on
D_eval*, the one subset both sides were scored on that iteration (line 15).

It is tempting to implement this as "return the pool's argmax", on the argument
that ``P_curr`` is itself the previous argmax so ``argmax(pool)`` telescopes into
``argmax(P_curr, P_new)``. That argument is **false** under subset evaluation,
because the pool's candidates are not scored on a common subset: Eq. 10 favors
B[M,1] / B[M,0] and never samples B[H,0], so each ``D_eval`` is systematically
easier than ``D``, and a candidate averaged over ``N`` such ids can outrank one
averaged over all of ``D`` without being better. A run built on that reasoning
climbed 0.435 -> 0.467 -> 0.600 -> 0.667 along its lineage while its actual
held-out score fell, because each step moved into an easier subset rather than a
better prompt.

The pointer therefore advances only through the policy's line 16 comparison,
which restricts both sides to the current ``D_eval`` so the sampling bias applies
to each equally and cancels. Rejected candidates stay in the pool inertly -- they
are never selected as a parent -- while still contributing to ``H``, which
Algorithm 1 line 19 requires.

Why not the existing selectors
------------------------------
* :class:`~gepa.strategies.candidate_selector.ParetoCandidateSelector` samples
  from the Pareto frontier; it is not an argmax.
* :class:`~gepa.strategies.candidate_selector.CurrentBestCandidateSelector` uses
  ``idxmax(state.program_full_scores_val_set)``, i.e. each candidate's average
  over *its own* evaluated ids -- exactly the non-comparable quantity described
  above.

This selector delegates to the policy's ``get_best_program``, which reports
``P_curr`` -- the variable line 17 assigns, advanced only by line 16's
comparison on a shared ``D_eval``.
"""

from __future__ import annotations

from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.eval_policy import EvaluationPolicy


class ApexCurrentBestSelector(CandidateSelector):
    """Return the pool's argmax, judged on shared coverage.

    Args:
        policy: The evaluation policy whose ``get_best_program`` defines "best".
            Pass the same :class:`~gepa.strategies.apex_eval_policy.ApexRankSensitivePolicy`
            instance given to the engine, so selection and reporting agree on
            which candidate is ``P_curr``.
    """

    def __init__(self, policy: EvaluationPolicy):
        self.policy = policy

    def select_candidate_idx(self, state: GEPAState) -> int:
        return self.policy.get_best_program(state)


__all__ = ["ApexCurrentBestSelector"]
