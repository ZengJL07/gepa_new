"""Render (candidate x train-example) score heatmaps for GEPA runs.

Rows are accepted candidates sorted by their aggregate validation score
(descending). Columns are train-example ids. A cell has three states:

  solved      score >= 0.5 on that train example
  failed      score <  0.5
  unevaluated never in any minibatch that this candidate was scored on

The matrix is sparse by construction: GEPA evaluates a candidate only on the
3-example minibatch that produced it (plus its parent's), never on the full
trainset. The sparsity pattern IS the sampler's fingerprint, so the panel
carries an aligned strip of per-column pull counts (n_k) underneath.

Outputs a self-contained HTML (hover tooltips, light+dark) and a PNG.
"""

import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "figures"

TAU = 0.5  # solved threshold, matches CapabilityTransferUCBSampling.tau

PANELS = [
    ("hmmt_baseline", "HMMT · stock GEPA (baseline sampler)"),
    ("hmmt_captransfer", "HMMT · capability-transfer UCB (ours)"),
    ("math500_captransfer_3", "MATH-500 · capability-transfer UCB (ours)"),
]

# Sequential blue ramp (reference palette). Heatmap = sequential encoding, so
# the near-surface step is allowed; "unevaluated" is surface + hairline instead.
LIGHT = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "solved": "#1c5cab",   # blue 550
    "failed": "#cde2fb",   # blue 100
    "bar": "#2a78d6",      # blue 450
    "border": "rgba(11,11,11,0.10)",
}
DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "solved": "#86b6ef",   # blue 250 (light-on-dark inversion)
    "failed": "#184f95",   # blue 600
    "bar": "#3987e5",
    "border": "rgba(255,255,255,0.10)",
}


def gini(xs: list[float]) -> float:
    """Gini coefficient of a non-negative allocation (0 = uniform)."""
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    s = sorted(xs)
    cum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(s))
    return cum / (n * sum(s))


def load_panel(name: str) -> dict:
    d = json.loads((DATA_DIR / f"{name}.json").read_text())
    n_train = d["n_train"]

    rows = sorted(d["candidates"], key=lambda c: (-c["val_avg"], c["idx"]))

    pulls = [0] * n_train
    for it in d["iterations"]:
        for t in it["tasks"]:
            for k in t["subsample_ids"] or []:
                if 0 <= int(k) < n_train:
                    pulls[int(k)] += 1

    obs = [0] * n_train
    solved = [0] * n_train
    for c in d["candidates"]:
        for k, v in c["train_scores"].items():
            ki = int(k)
            obs[ki] += 1
            if v >= TAU:
                solved[ki] += 1

    d["_rows"] = rows
    d["_pulls"] = pulls
    d["_obs"] = obs
    d["_solved"] = solved
    d["_gini"] = gini([float(p) for p in pulls])
    mean = sum(pulls) / n_train
    var = sum((p - mean) ** 2 for p in pulls) / n_train
    d["_cv"] = math.sqrt(var) / mean if mean else 0.0
    return d
