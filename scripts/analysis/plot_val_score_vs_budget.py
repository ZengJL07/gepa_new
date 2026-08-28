#!/usr/bin/env python
"""Plot validation score against metric-call budget for GEPA runs.

Compares the capability-transfer sampling strategy against stock GEPA on two
datasets. Reads ``gepa_state.bin`` only -- no API calls, no re-evaluation.

Two series per figure:

- **Best-so-far** (step line): running argmax of the per-candidate aggregate
  validation score, plotted at the budget each candidate was discovered at.
  This is the prompt GEPA would hand you if stopped at budget X.
- **Individual candidates** (scatter): every accepted candidate, showing the
  spread the step line hides.

Both come from state arrays that are maintained in lockstep with
``program_candidates``:

- ``num_metric_calls_by_discovery[i]`` -- rollouts spent when candidate *i* was
  found (x axis).
- ``prog_candidate_val_subscores[i]`` -- that candidate's per-example validation
  scores; the aggregate is their mean (y axis).

Usage::

    .venv/bin/python scripts/analysis/plot_val_score_vs_budget.py
    .venv/bin/python scripts/analysis/plot_val_score_vs_budget.py --out-dir /tmp/figs
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from gepa.core.state import GEPAState

# --- Palette (dataviz reference instance, light mode) -----------------------
# Categorical slots 1 and 2. This adjacent pair is documented as clearing every
# hard gate (CVD and normal-vision separation) in both light and dark modes.
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

TEST_ROOT = REPO_ROOT / "examples" / "aime_math" / "test"


@dataclass(frozen=True)
class RunSpec:
    """One optimization run to plot."""

    label: str
    run_dir: Path


@dataclass(frozen=True)
class FigureSpec:
    """One output figure: a dataset and the runs compared on it."""

    slug: str
    dataset: str
    subtitle: str
    runs: tuple[RunSpec, ...]


FIGURES = (
    FigureSpec(
        slug="math500_captransfer3_vs_baseline",
        dataset="MATH-500",
        subtitle="45-example validation split, deepseek-v4-flash @ t=0.2, seed 42",
        runs=(
            RunSpec(
                "Capability transfer",
                TEST_ROOT / "math500_formal/gepa_captransfer_3/t_0.2/run_seed_42",
            ),
            RunSpec(
                "Baseline GEPA",
                TEST_ROOT / "math500_formal/gepa_baseline/t_0.2_run_seed_42_long",
            ),
        ),
    ),
    FigureSpec(
        slug="hmmt_captransfer_vs_baseline",
        dataset="HMMT",
        subtitle="33-example validation split (hmmt_feb_2026), deepseek-v4-flash @ t=0.2, seed 42",
        runs=(
            RunSpec(
                "Capability transfer",
                TEST_ROOT / "hmmt_formal/gepa_captransfer/t_0.2/run_seed_42",
            ),
            RunSpec(
                "Baseline GEPA",
                TEST_ROOT / "hmmt_formal/gepa_baseline/t_0.2/run_seed_42",
            ),
        ),
    ),
)

@dataclass
class RunCurve:
    """Extracted plot data for one run."""

    label: str
    budgets: list[int]
    scores: list[float]
    total_budget: int
    valset_size: int
    #: (budget, score) of each point where the running max improved.
    step_budgets: list[int]
    step_scores: list[float]


def load_curve(spec: RunSpec) -> RunCurve:
    """Read a run's candidate scores and discovery budgets out of gepa_state.bin."""
    if not (spec.run_dir / "gepa_state.bin").exists():
        raise FileNotFoundError(f"No gepa_state.bin under {spec.run_dir}")

    state = GEPAState.load(str(spec.run_dir))
    scores = list(state.program_full_scores_val_set)
    budgets = list(state.num_metric_calls_by_discovery)
    if len(scores) != len(budgets):
        raise ValueError(
            f"{spec.run_dir}: {len(scores)} candidate scores vs {len(budgets)} discovery counts"
        )

    coverage = {len(sub) for sub in state.prog_candidate_val_subscores}
    if len(coverage) != 1:
        # Every candidate here gets a full valset eval (FullEvaluationPolicy), so
        # a mixed coverage would mean the aggregates are not comparable.
        raise ValueError(f"{spec.run_dir}: uneven valset coverage {sorted(coverage)}")

    step_budgets: list[int] = []
    step_scores: list[float] = []
    best = float("-inf")
    for budget, score in zip(budgets, scores, strict=True):
        if score > best:
            best = score
            step_budgets.append(budget)
            step_scores.append(score)

    return RunCurve(
        label=spec.label,
        budgets=budgets,
        scores=scores,
        total_budget=state.total_num_evals,
        valset_size=coverage.pop(),
        step_budgets=step_budgets,
        step_scores=step_scores,
    )


def render(fig_spec: FigureSpec, curves: list[RunCurve], out_path: Path) -> None:
    """Draw one comparison figure and write it to ``out_path``."""
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for curve, color in zip(curves, SERIES, strict=False):
        # Extend the final step to the run's full budget so the line shows how
        # long that score was the incumbent, not just where it was discovered.
        xs = [*curve.step_budgets, curve.total_budget]
        ys = [*curve.step_scores, curve.step_scores[-1]]
        ax.step(xs, ys, where="post", color=color, linewidth=2.0,
                solid_joinstyle="round", solid_capstyle="round", zorder=3)
        # Every accepted candidate: the spread behind the step line.
        ax.scatter(curve.budgets, curve.scores, s=26, color=color, alpha=0.30,
                   linewidths=0, zorder=2)
        # Improvement points get a surface ring so they stay legible on overlap.
        ax.scatter(curve.step_budgets, curve.step_scores, s=64, color=color,
                   edgecolors=SURFACE, linewidths=2.0, zorder=4)

    # Direct-label the final score of each series at the right edge.
    finals = sorted(
        ((c.step_scores[-1], c.total_budget, col) for c, col in zip(curves, SERIES, strict=False)),
        reverse=True,
    )
    span = max(c.total_budget for c in curves)
    for score, budget, _color in finals:
        ax.annotate(
            f"{score:.3f}",
            xy=(budget, score),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=INK_PRIMARY,
        )

    seed_scores = {round(c.scores[0], 6) for c in curves}
    seed_note = (
        f"seed prompt {next(iter(seed_scores)):.3f}"
        if len(seed_scores) == 1
        else "seed prompt " + " / ".join(f"{s:.3f}" for s in sorted(seed_scores, reverse=True))
    )

    ax.set_title(
        f"{fig_spec.dataset}: validation score vs rollout budget",
        fontsize=13.5, fontweight="bold", color=INK_PRIMARY, loc="left", pad=14,
    )
    ax.text(
        0.0, 1.015, f"{fig_spec.subtitle} · {seed_note}",
        transform=ax.transAxes, fontsize=9.5, color=INK_SECONDARY, ha="left", va="bottom",
    )
    ax.set_xlabel("Metric calls (rollouts) spent", fontsize=10.5, color=INK_SECONDARY, labelpad=8)
    ax.set_ylabel(
        f"Validation accuracy ({curves[0].valset_size} examples)",
        fontsize=10.5, color=INK_SECONDARY, labelpad=8,
    )

    ax.set_xlim(-span * 0.02, span * 1.10)
    lo = min(min(c.scores) for c in curves)
    hi = max(max(c.scores) for c in curves)
    pad = max((hi - lo) * 0.12, 0.02)
    ax.set_ylim(lo - pad, hi + pad)

    ax.grid(True, axis="y", color=GRIDLINE, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9.5, length=0)
    for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lbl.set_fontfamily("sans-serif")

    handles = [
        Line2D([], [], color=col, linewidth=2.0, marker="o", markersize=7,
               markeredgecolor=SURFACE, markeredgewidth=2.0, label=c.label)
        for c, col in zip(curves, SERIES, strict=False)
    ]
    handles.append(
        Line2D([], [], color=INK_MUTED, linewidth=0, marker="o", markersize=5,
               alpha=0.45, label="individual candidate")
    )
    legend = ax.legend(
        handles=handles, loc="lower right", frameon=True, fontsize=9.5,
        facecolor=SURFACE, edgecolor=GRIDLINE, borderpad=0.8, labelspacing=0.7,
    )
    legend.get_frame().set_linewidth(1.0)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_table(fig_spec: FigureSpec, curves: list[RunCurve], out_path: Path) -> None:
    """Write the table-view twin: every plotted point, machine-readable."""
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "run", "candidate_idx", "metric_calls", "val_score", "is_best_so_far"]
        )
        for curve in curves:
            improvements = set(zip(curve.step_budgets, curve.step_scores, strict=True))
            for idx, (budget, score) in enumerate(zip(curve.budgets, curve.scores, strict=True)):
                writer.writerow([
                    fig_spec.dataset, curve.label, idx, budget,
                    f"{score:.6f}", int((budget, score) in improvements),
                ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "val_score_vs_budget",
        help="directory for the PNG + CSV output (default: results/val_score_vs_budget)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fig_spec in FIGURES:
        curves = [load_curve(run) for run in fig_spec.runs]
        png = args.out_dir / f"{fig_spec.slug}.png"
        csv_path = args.out_dir / f"{fig_spec.slug}.csv"
        render(fig_spec, curves, png)
        write_table(fig_spec, curves, csv_path)

        print(f"{fig_spec.dataset}:")
        for curve in curves:
            print(
                f"  {curve.label:22s} seed {curve.scores[0]:.4f} -> best {max(curve.scores):.4f}"
                f"  ({len(curve.scores)} candidates, {len(curve.step_scores)} improvements,"
                f" {curve.total_budget} rollouts)"
            )
        print(f"  wrote {png.relative_to(REPO_ROOT)}")
        print(f"  wrote {csv_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
