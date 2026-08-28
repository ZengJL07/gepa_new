"""Extract a (candidate x train-example) score matrix from a GEPA run_dir.

Train-example scores are not kept in ``gepa_state.bin`` (which stores only
valset subscores). They live in ``fitness_cache/<cand16>_<example16>.pkl``,
written by OptimizeAnythingAdapter on every cache miss. This script re-derives
the split (so example hashes can be mapped back to train ids), then joins the
cache against the accepted-candidate list.

Emits one JSON per run to scripts/analysis/data/<name>.json.
"""

import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.aime_math.utils import load_math_dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "data"


def example_hash(example) -> str:
    """Mirror OptimizeAnythingAdapter._example_hash."""
    return hashlib.sha256(json.dumps(example, sort_keys=True, default=str).encode()).hexdigest()[:16]


def candidate_hash(candidate: dict[str, str]) -> str:
    """Mirror OptimizeAnythingAdapter._candidate_hash."""
    return hashlib.sha256(json.dumps(sorted(candidate.items())).encode()).hexdigest()[:16]


def load_state(run_dir: Path):
    sys.path.insert(0, str(REPO / "src"))
    from gepa.core.state import GEPAState

    return GEPAState.load(str(run_dir))


def parse_rejected_proposals(run_dir: Path) -> list[dict]:
    """Recover proposal texts + verdicts from run_log.txt.

    write_agent_state was off for these runs, so the trace holds no proposal
    text. The log is the only record: a run of "Proposed new text for X: ..."
    lines followed, in the same order, by accept/skip verdict lines.
    """
    log = (run_dir / "run_log.txt").read_text().splitlines()
    prop_re = re.compile(r"^Iteration (\d+): Proposed new text for (\S+): (.*)$")
    out = []
    for line in log:
        m = prop_re.match(line)
        if m:
            out.append({"iteration": int(m.group(1)), "component": m.group(2), "text": m.group(3)})
    return out


def build(run_dir: Path, dataset: str, sizes, seed: int, name: str) -> dict:
    train, val, _test = load_math_dataset(name=dataset, sizes=sizes, seed=seed)
    train_by_hash = {example_hash(e): i for i, e in enumerate(train)}
    val_by_hash = {example_hash(e): i for i, e in enumerate(val)}
    assert len(train_by_hash) == len(train), "train example hash collision"
    assert len(val_by_hash) == len(val), "val example hash collision"

    state = load_state(run_dir)
    candidates = state.program_candidates
    cand_hashes = [candidate_hash(c) for c in candidates]
    hash_to_cand = {h: i for i, h in enumerate(cand_hashes)}

    # fitness_cache: (candidate, example) -> score. Covers accepted candidates
    # AND rejected proposals, train AND val examples.
    train_scores: dict[int, dict[int, float]] = {}
    other_cand_train: dict[str, dict[int, float]] = {}
    n_cache = n_train_cells = n_val_cells = 0
    for pkl in (run_dir / "fitness_cache").glob("*.pkl"):
        n_cache += 1
        stem = pkl.stem
        ch, eh = stem.split("_")
        with open(pkl, "rb") as f:
            payload = pickle.load(f)
        score = float(payload["result"][0])
        if eh in train_by_hash:
            n_train_cells += 1
            tid = train_by_hash[eh]
            if ch in hash_to_cand:
                train_scores.setdefault(hash_to_cand[ch], {})[tid] = score
            else:
                other_cand_train.setdefault(ch, {})[tid] = score
        elif eh in val_by_hash:
            n_val_cells += 1

    val_avg = state.program_full_scores_val_set

    # Per-iteration minibatch record (which train ids each proposal task saw).
    iterations = []
    for entry in state.full_program_trace:
        iterations.append(
            {
                "i": entry.get("i"),
                "iteration_id": entry.get("iteration_id"),
                "n_tasks": entry.get("n_tasks"),
                "accepted_candidate_idxs": entry.get("new_program_indices") or [],
                "tasks": [
                    {
                        "parent_idx": t.get("parent_idx"),
                        "subsample_ids": t.get("subsample_ids"),
                        "subsample_scores": t.get("subsample_scores"),
                        "new_subsample_scores": t.get("new_subsample_scores"),
                    }
                    for t in (entry.get("tasks") or [])
                ],
            }
        )

    return {
        "name": name,
        "run_dir": str(run_dir.relative_to(REPO)),
        "dataset": dataset,
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "n_iterations": state.i + 1,
        "n_candidates": len(candidates),
        "total_metric_calls": state.total_num_evals,
        "candidates": [
            {
                "idx": i,
                "val_avg": val_avg[i],
                "parents": [p for p in state.parent_program_for_candidate[i]],
                "iteration_id": state.iteration_ids_by_candidate_idx[i],
                "metric_calls_at_discovery": state.num_metric_calls_by_discovery[i],
                "train_scores": {str(k): v for k, v in sorted(train_scores.get(i, {}).items())},
                "val_scores": {str(k): v for k, v in sorted(state.prog_candidate_val_subscores[i].items())},
                "text": candidates[i].get("current_candidate", next(iter(candidates[i].values()), "")),
            }
            for i in range(len(candidates))
        ],
        "rejected_proposal_train_scores": {
            ch: {str(k): v for k, v in sorted(d.items())} for ch, d in other_cand_train.items()
        },
        "iterations": iterations,
        "cache_stats": {
            "files": n_cache,
            "train_cells": n_train_cells,
            "val_cells": n_val_cells,
            "distinct_candidate_hashes": len({p.stem.split("_")[0] for p in (run_dir / "fitness_cache").glob("*.pkl")}),
        },
        "log_proposals": parse_rejected_proposals(run_dir),
    }


RUNS = [
    {
        "name": "math500_captransfer_3",
        "run_dir": REPO / "examples/aime_math/test/math500_formal/gepa_captransfer_3/t_0.2/run_seed_42",
        "dataset": "math500",
        "sizes": (40, 45, 100),
        "seed": 42,
    },
    {
        "name": "hmmt_captransfer",
        "run_dir": REPO / "examples/aime_math/test/hmmt_formal/gepa_captransfer/t_0.2/run_seed_42",
        "dataset": "hmmt",
        "sizes": (None, None, None),
        "seed": 42,
    },
    {
        "name": "hmmt_baseline",
        "run_dir": REPO / "examples/aime_math/test/hmmt_formal/gepa_baseline/t_0.2/run_seed_42",
        "dataset": "hmmt",
        "sizes": (None, None, None),
        "seed": 42,
    },
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in RUNS:
        data = build(spec["run_dir"], spec["dataset"], spec["sizes"], spec["seed"], spec["name"])
        out = OUT_DIR / f"{spec['name']}.json"
        out.write_text(json.dumps(data, indent=1))
        filled = sum(len(c["train_scores"]) for c in data["candidates"])
        print(
            f"{data['name']}: {data['n_candidates']} cands x {data['n_train']} train "
            f"= {data['n_candidates'] * data['n_train']} cells, {filled} observed "
            f"({filled / (data['n_candidates'] * data['n_train']):.1%}); "
            f"rejected-proposal hashes with train cells: {len(data['rejected_proposal_train_scores'])}; "
            f"cache {data['cache_stats']}"
        )


if __name__ == "__main__":
    main()
