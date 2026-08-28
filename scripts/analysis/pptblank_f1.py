"""Post-hoc F1 for a pptblank run, plus F1-based candidate re-ranking.

Why this exists: GEPA aggregates per-example scores by averaging (see
``GEPAState.program_full_scores_val_set``), and F1 = 2TP/(2TP+FP+FN) is not
expressible as a mean of per-example scores — true negatives never enter the
denominator, and no single example knows the corpus-wide counts. So the loop
optimizes plain 0/1 correctness and F1 is reconstructed here, afterwards.

The reconstruction is exact rather than approximate: ``prog_candidate_val_subscores``
records each candidate's score on each validation example, scores are 0/1, and
``prepare_pptblank.py`` removed every ``either`` slide from val — so each val
example is unambiguously positive or negative, and score==1 on a positive example
is exactly one true positive.

Joining on position, not on id: GEPA's ``ListDataLoader`` keys validation examples
by their index in the valset list, so the subscore keys are ints. This script
reloads the valset through the same ``load_math_dataset`` call the run used and
maps index -> example -> gold. That makes the join depend on valset ordering, so
the split sizes must match the run's; pass the same ``AIME_VAL_K`` (and seed) the
run used, and the coverage assertion below will catch a mismatch.

Usage::

    python scripts/analysis/pptblank_f1.py <run_dir>
    AIME_VAL_K=6 python scripts/analysis/pptblank_f1.py /tmp/pptblank_smoke
"""

import argparse
import json
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def f1_from(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, f1); all zero when nothing was predicted positive."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return precision, recall, f1


def confusion(subscores: dict, gold_by_index: dict) -> dict:
    """Confusion counts for one candidate over the val examples it was scored on.

    ``either``-gold examples are excluded: they score 1.0 unconditionally, so
    counting them would inflate TP with slides that cannot be gotten wrong. They
    should not appear in val at all (the preparation script drops them), but the
    guard keeps this usable on a test-split evaluation too.
    """
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "skipped": 0, "unknown": 0}
    for key, score in subscores.items():
        gold = gold_by_index.get(int(key))
        if gold is None:
            counts["unknown"] += 1
            continue
        if gold == "either":
            counts["skipped"] += 1
            continue
        correct = score >= 0.5
        if gold == "yes":
            counts["tp" if correct else "fn"] += 1
        else:
            counts["tn" if correct else "fp"] += 1
    return counts


def load_state(run_dir: str) -> dict:
    """Load the run's serialized GEPAState (a plain dict inside gepa_state.bin)."""
    path = os.path.join(run_dir, "gepa_state.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Expected a GEPA run directory; it contains: "
            f"{sorted(os.listdir(run_dir))[:20] if os.path.isdir(run_dir) else 'not a directory'}"
        )
    with open(path, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, dict) or "prog_candidate_val_subscores" not in state:
        raise ValueError(
            f"{path} does not look like a GEPA state dict "
            f"(type={type(state).__name__}, keys={sorted(state)[:10] if isinstance(state, dict) else 'n/a'})"
        )
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument(
        "--dataset", default=os.environ.get("AIME_DATASET", "pptblank"),
        help="dataset name the run used (default: $AIME_DATASET or pptblank)",
    )
    args = ap.parse_args()

    os.environ.setdefault("AIME_DATASET", args.dataset)
    from examples.aime_math.utils import load_math_dataset, split_sizes_from_env

    state = load_state(args.run_dir)
    subscores_list = state["prog_candidate_val_subscores"]
    candidates = state.get("program_candidates") or []

    # Rebuild the valset exactly as the run did, so index -> gold is the same map.
    seed = int(os.environ.get("AIME_SEED", 0))
    _, valset, _ = load_math_dataset(name=args.dataset, sizes=split_sizes_from_env(), seed=seed)
    gold_by_index = {i: ex.answer for i, ex in enumerate(valset)}
    id_by_index = {i: ex.id for i, ex in enumerate(valset)}

    # Every key a candidate was scored on must exist in the rebuilt valset,
    # otherwise the join is against a different split and the numbers are junk.
    seen = {int(k) for s in subscores_list for k in s}
    missing = sorted(seen - set(gold_by_index))
    if missing:
        print(
            f"ERROR: the run scored val indices {missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"but the rebuilt valset has only {len(gold_by_index)} examples.\n"
            f"Re-run with the same AIME_VAL_K / AIME_TRIM_K / AIME_SEED the run used.",
            file=sys.stderr,
        )
        return 1

    n_val = max((len(s) for s in subscores_list), default=0)
    rows = []
    for idx, subscores in enumerate(subscores_list):
        c = confusion(subscores, gold_by_index)
        p, r, f1 = f1_from(c["tp"], c["fp"], c["fn"])
        agg = sum(subscores.values()) / len(subscores) if subscores else float("nan")
        rows.append({
            "idx": idx, "accuracy": agg, "precision": p, "recall": r, "f1": f1,
            "coverage": len(subscores), **c,
        })

    print(f"run:        {args.run_dir}")
    print(f"dataset:    {args.dataset}   valset rebuilt with {len(valset)} examples")
    print(f"candidates: {len(rows)}      max val coverage: {n_val}")

    partial = [r for r in rows if r["coverage"] < n_val]
    if partial:
        print(
            f"WARNING: {len(partial)} candidate(s) scored on a SUBSET of val (min coverage "
            f"{min(r['coverage'] for r in rows)}). F1 is not comparable across candidates with "
            f"different coverage — expected under subset evaluation policies like "
            f"ApexRankSensitivePolicy, and also normal early in a run before a candidate has "
            f"been fully evaluated."
        )
    if any(r["skipped"] for r in rows):
        print("note: 'either'-gold examples excluded from F1 (correct by construction).")
    print()

    hdr = (f'{"cand":>4s} {"acc":>6s} {"F1":>6s} {"prec":>6s} {"recall":>7s} '
           f'{"TP":>3s} {"FP":>3s} {"FN":>3s} {"TN":>3s} {"cov":>4s}')
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["f1"]):
        print(f'{r["idx"]:4d} {r["accuracy"]:6.3f} {r["f1"]:6.3f} {r["precision"]:6.3f} '
              f'{r["recall"]:7.3f} {r["tp"]:3d} {r["fp"]:3d} {r["fn"]:3d} {r["tn"]:3d} '
              f'{r["coverage"]:4d}')

    best_f1 = max(rows, key=lambda r: r["f1"])
    best_acc = max(rows, key=lambda r: r["accuracy"])
    print()
    print(f'argmax F1       = candidate {best_f1["idx"]}  (F1={best_f1["f1"]:.3f}, acc={best_f1["accuracy"]:.3f})')
    print(f'argmax accuracy = candidate {best_acc["idx"]}  (F1={best_acc["f1"]:.3f}, acc={best_acc["accuracy"]:.3f})')
    if best_f1["idx"] != best_acc["idx"]:
        print(f'  DISAGREE: accuracy-selection (what GEPAResult.best_idx uses) costs '
              f'{best_f1["f1"] - best_acc["f1"]:+.3f} F1 versus selecting on F1.')
    else:
        print("  agree: accuracy-selection and F1-selection pick the same candidate.")

    d = best_f1
    P, N = d["tp"] + d["fn"], d["fp"] + d["tn"]
    if P:
        _, _, all_yes = f1_from(P, N, 0)
        print()
        print(f"reference on these {P + N} decisive val examples ({P} pos / {N} neg): "
              f"all-yes F1 = {all_yes:.3f}, all-no F1 = 0.000")

    out = {
        "run_dir": args.run_dir, "dataset": args.dataset,
        "rows": rows, "argmax_f1": best_f1["idx"], "argmax_accuracy": best_acc["idx"],
        "val_index_to_id": id_by_index,
        "best_f1_candidate": candidates[best_f1["idx"]] if best_f1["idx"] < len(candidates) else None,
    }
    out_path = os.path.join(args.run_dir, "pptblank_f1.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
