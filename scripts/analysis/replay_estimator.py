"""Replay CapabilityTransferUCBSampling to recover its estimator matrices.

The estimator's state (A, B, obs, fix, n_k) is never persisted: it lives in the
strategy instance, and adapter_state is empty for these runs. So we rebuild it
offline from run_log.json's per-task records by driving the REAL strategy object
(imported from gepa, not reimplemented) through the run's history in order:

  for each iteration:
      for each task:  emit(parent, minibatch)   # n_k, total_emits, _pending
      observe_proposals(all tasks)              # obs, fix  <- train before/after
      reconcile(accepted candidates)            # A, B      <- val before/after

Fidelity check: at each iteration we ask the replayed estimator which minibatch
it WOULD pick for the parent the log recorded, and compare to what the log says
it did pick. A faithful replay reproduces them (the strategy's only randomness
is a seeded tie-break RNG inside _select_batch).

Emits scripts/analysis/data/<name>.estimator.json holding, for the final state:
  - mu[j][k]      Beta-smoothed capability-transfer utility
  - U[k]          usability gate
  - n_k[k]        pull counts
  - per-candidate value(k) = U[k]^gamma * mean_{j in gaps(c)} mu[j][k]
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gepa.strategies.capability_transfer_sampling import CapabilityTransferUCBSampling  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

# Defaults from scripts/formal/run_*_gepa_captransfer.sh (all GEPA_CT_* unset).
CT_KWARGS = dict(
    n=5,
    minibatch_size=3,
    tau=0.5,
    alpha=1.0,
    beta=1.0,
    exploration_weight=0.2,
    cold_start_bonus=0.2,
    usability_weight=1.0,
    alpha_u=1.0,
    beta_u=1.0,
    seed=42,
)


class _StateView:
    """Minimal GEPAState stand-in: the strategy only reads these two fields."""

    def __init__(self, val_subscores, parents):
        self.prog_candidate_val_subscores = val_subscores
        self.parent_program_for_candidate = parents

    @property
    def program_candidates(self):
        return self.prog_candidate_val_subscores


class _Proposal:
    __slots__ = ("subsample_indices", "subsample_scores_before", "subsample_scores_after")

    def __init__(self, ids, before, after):
        self.subsample_indices = ids
        self.subsample_scores_before = before
        self.subsample_scores_after = after


def replay(name: str) -> dict:
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    n_train = d["n_train"]
    train_ids = list(range(n_train))

    by_idx = {c["idx"]: c for c in d["candidates"]}
    n_cands = len(by_idx)

    ct = CapabilityTransferUCBSampling(**CT_KWARGS)

    # Candidates become visible to the estimator only as they are accepted, so
    # the state view grows over the replay exactly as it did during the run.
    val_subscores: list[dict[int, float]] = []
    parents: list[list] = []

    def push(idx: int):
        c = by_idx[idx]
        val_subscores.append({int(k): v for k, v in c["val_scores"].items()})
        parents.append(list(c["parents"]))

    push(0)  # seed
    state = _StateView(val_subscores, parents)

    matches = mismatches = 0
    per_iteration = []

    for it in d["iterations"]:
        tasks = it["tasks"] or []
        if not tasks:
            continue

        # Fold the previous iteration's accepted candidates into A/B, exactly as
        # sample_tasks() does at its top.
        ct._reconcile(state)

        # Predict-then-emit: for each task, score columns for the logged parent
        # and check the top-minibatch_size selection against the logged ids. The
        # emit bookkeeping (n_k, total_emits, _pending) must mirror the real
        # sample_tasks loop, including the within-iteration exclusion set.
        used: set[int] = set()
        it_rec = {"i": it["i"], "tasks": []}
        for t in tasks:
            parent_idx = t["parent_idx"]
            ids = [int(x) for x in (t["subsample_ids"] or [])]
            if parent_idx is None or parent_idx >= len(val_subscores):
                # Parent not yet visible (accepted later in this same iteration);
                # replay the emit without a prediction.
                predicted = None
            else:
                scores = ct._score_columns(state, parent_idx, train_ids)
                predicted = ct._select_batch(scores, train_ids, set(used))
                if sorted(predicted) == sorted(ids):
                    matches += 1
                else:
                    mismatches += 1
            used.update(ids)

            ct.total_emits += 1
            for k in ids:
                ct.n_k[k] += 1
            ct._pending[parent_idx].append(tuple(ids))

            it_rec["tasks"].append(
                {
                    "parent_idx": parent_idx,
                    "logged": ids,
                    "predicted": predicted,
                    "match": None if predicted is None else sorted(predicted) == sorted(ids),
                }
            )

        # Usability from ALL evaluated proposals (accepted or not).
        ct.observe_proposals(
            [
                _Proposal(
                    [int(x) for x in (t["subsample_ids"] or [])],
                    list(t["subsample_scores"] or []),
                    list(t["new_subsample_scores"] or []),
                )
                for t in tasks
                if t["subsample_ids"] and t["subsample_scores"] and t["new_subsample_scores"]
            ]
        )

        for idx in it["accepted_candidate_idxs"]:
            if idx in by_idx and idx >= len(val_subscores):
                push(idx)

        per_iteration.append(it_rec)

    ct._reconcile(state)

    # ---- final estimator state -------------------------------------------
    val_ids = sorted({int(k) for c in d["candidates"] for k in c["val_scores"]})
    mu = {str(j): {str(k): ct._mu(j, k) for k in train_ids} for j in val_ids}
    usability = {str(k): ct._usability(k) for k in train_ids}
    pulls = {str(k): ct.n_k[k] for k in train_ids}
    a_counts = {f"{j}|{k}": v for (j, k), v in ct.A.items() if v}
    b_counts = {f"{j}|{k}": v for (j, k), v in ct.B.items() if v}

    # value(k) per candidate: the estimated improvement train example k offers
    # to candidate c, masked to the val questions c currently fails.
    cand_value = {}
    for idx in range(n_cands):
        if idx >= len(val_subscores):
            continue
        vs = val_subscores[idx]
        gaps = [j for j, s in vs.items() if s < ct.tau]
        row = {}
        for k in train_ids:
            rel = sum(ct._mu(j, k) for j in gaps) / len(gaps) if gaps else 0.0
            gate = ct._usability(k) ** ct.usability_weight if ct.usability_weight > 0 else 1.0
            row[str(k)] = gate * rel
        cand_value[str(idx)] = {"n_gaps": len(gaps), "value": row}

    return {
        "name": name,
        "dataset": d["dataset"],
        "n_train": n_train,
        "n_val": d["n_val"],
        "n_candidates": n_cands,
        "ct_kwargs": CT_KWARGS,
        "fidelity": {
            "predicted_minibatches": matches + mismatches,
            "matches": matches,
            "mismatches": mismatches,
            "rate": matches / (matches + mismatches) if (matches + mismatches) else None,
        },
        "total_emits": ct.total_emits,
        "mu": mu,
        "usability": usability,
        "pulls": pulls,
        "A": a_counts,
        "B": b_counts,
        "candidate_value": cand_value,
        "per_iteration": per_iteration,
    }


def main():
    for name in ["hmmt_captransfer", "math500_captransfer_3"]:
        out = replay(name)
        (DATA_DIR / f"{name}.estimator.json").write_text(json.dumps(out, indent=1))
        f = out["fidelity"]
        print(
            f"{name}: replay fidelity {f['matches']}/{f['predicted_minibatches']} "
            f"({(f['rate'] or 0):.1%}) | emits={out['total_emits']} "
            f"| A cells={len(out['A'])} B cells={len(out['B'])}"
        )


if __name__ == "__main__":
    main()
