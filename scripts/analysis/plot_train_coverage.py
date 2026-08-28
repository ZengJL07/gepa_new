"""Plot training-set coverage for GEPA runs: baseline vs capability-transfer sampling.

Standalone — reads only ``run_log.json`` from each run dir (the persisted
``full_program_trace``). Every iteration's trace entry carries a ``tasks`` list,
one record per parallel proposal, each with the ``subsample_ids`` of the training
minibatch that proposal was built from. Those ids are ``ListDataLoader``
indices into the trainset produced by ``load_math_dataset(name, sizes, seed)``,
so coverage is computable without re-running any model.

Two views per dataset:

* **Coverage curve** — cumulative distinct training examples touched, as a
  function of cumulative minibatch slots consumed. Answers "how fast does the
  strategy reach the whole trainset".
* **Visit profile** — per-example sample counts, sorted descending. Answers
  "once covered, how evenly is the trainset revisited". Gini + max annotate it.

Outputs PNG (light + dark) per dataset plus a CSV table view, so no value is
reachable only through the figure.

Dead iterations are MASKED. A run dir can accumulate multiple process launches
(the logger appends and ``gepa_state.bin`` is resumed), and a launch whose
reflection LM runs out of credit keeps looping: it pays for each parent's
minibatch evaluation, fails to propose, and produces nothing until the budget
stops it. Such iterations — every task lacking ``new_subsample_scores`` — carry
no sampling-quality signal, so ``load_coverage`` drops them from the curve and
both profiles, and reports how many were masked. Masking is essential here: the
hmmt baseline's 11 dead iterations would otherwise inflate its visit counts by
165 slots and flatten its coverage curve into a fake plateau.

Two caveats masking cannot repair, both reported rather than hidden:

* Dead iterations still consumed rollout budget on parent evaluations, so the
  masked run did less useful work per rollout than its slot count suggests.
* ``state.i`` advanced through them, and the batch sampler keys its chunk
  offset on ``state.i`` — so the minibatch sequence AFTER a dead span is not
  the sequence a clean run would have produced. The sampler is also not
  persisted across a restart (each launch builds a fresh
  ``EpochShuffledBatchSampler``), so relaunch boundaries reshuffle it.

Usage:
    uv run python scripts/analysis/plot_train_coverage.py
    uv run python scripts/analysis/plot_train_coverage.py --outdir /tmp/figs
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "examples" / "aime_math" / "test"

# --- Palette (dataviz reference instance, slots 1-2) -----------------------
# Blue = stock GEPA, orange = capability transfer. Colour follows the strategy,
# never its rank, so the two hues are fixed per strategy across every figure.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "baseline": "#2a78d6",
        "captransfer": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "baseline": "#3987e5",
        "captransfer": "#d95926",
    },
}

FONTS = ["DejaVu Sans", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif"]


@dataclass
class RunSpec:
    """One run to plot: where it lives, what it is, how big its trainset was."""

    label: str
    run_dir: Path
    strategy: str  # "baseline" | "captransfer" -> picks the hue
    trainset_size: int
    budget: int  # AIME_MAX_METRIC_CALLS from the launch script
    note: str = ""


@dataclass
class Coverage:
    """Coverage series derived from one run's trace."""

    spec: RunSpec
    slots: list[int] = field(default_factory=list)  # cumulative minibatch slots
    distinct: list[int] = field(default_factory=list)  # cumulative distinct ids
    counts: Counter = field(default_factory=Counter)  # id -> times sampled
    # id -> times it sat in a minibatch whose proposal was actually generated
    # AND evaluated (reflection can fail, leaving a sampled example with no
    # child to judge — see ``n_failed_tasks``).
    attempts: Counter = field(default_factory=Counter)
    # id -> times its minibatch produced a child that BEAT the parent on that
    # minibatch, i.e. successfully induced a new accepted candidate.
    induced: Counter = field(default_factory=Counter)
    n_iterations: int = 0  # LIVE iterations (dead ones are masked out)
    n_proposals: int = 0
    n_dead_iterations: int = 0  # masked: every task failed to propose
    n_masked_tasks: int = 0
    n_masked_slots: int = 0

    @property
    def total_slots(self) -> int:
        return sum(self.counts.values())

    @property
    def n_distinct(self) -> int:
        return len(self.counts)

    @property
    def coverage_frac(self) -> float:
        return self.n_distinct / self.spec.trainset_size

    @property
    def visit_profile(self) -> list[int]:
        """Per-example visit counts over the WHOLE trainset, ranked descending.

        Never-sampled examples are explicit zeros — the untouched tail is the
        point of the chart, so it must not be silently dropped. Uses
        :meth:`profile_order` so the induced series aligns rank-for-rank.
        """
        return [self.counts[i] for i in self.profile_order()]

    def profile_order(self) -> list[int]:
        """Example ids ranked by visits desc, then inductions desc, then id.

        One shared ordering for both series drawn in the right-hand panel, so
        the induced marks sit under the visit step they belong to.
        """
        return sorted(
            range(self.spec.trainset_size),
            key=lambda i: (-self.counts[i], -self.induced[i], i),
        )

    def induced_profile(self) -> list[int]:
        """Induction counts in :meth:`profile_order`."""
        return [self.induced[i] for i in self.profile_order()]

    def attempts_profile(self) -> list[int]:
        """Evaluated-attempt counts in :meth:`profile_order`."""
        return [self.attempts[i] for i in self.profile_order()]

    @property
    def total_induced(self) -> int:
        return sum(self.induced.values())

    @property
    def total_attempts(self) -> int:
        return sum(self.attempts.values())

    @property
    def n_induced_examples(self) -> int:
        return sum(1 for v in self.induced.values() if v > 0)

    @property
    def induction_rate(self) -> float:
        """Share of evaluated attempts that produced an accepted candidate."""
        return self.total_induced / self.total_attempts if self.total_attempts else 0.0

    @property
    def slots_to_full_coverage(self) -> int | None:
        """Slots consumed when the last unseen example was first sampled."""
        target = self.spec.trainset_size
        for slot, dist in zip(self.slots, self.distinct, strict=True):
            if dist >= target:
                return slot
        return None

    @property
    def gini(self) -> float:
        """Gini of the visit profile: 0 = perfectly even, 1 = all on one example."""
        xs = sorted(self.visit_profile)
        n = len(xs)
        total = sum(xs)
        if n == 0 or total == 0:
            return 0.0
        weighted = sum((i + 1) * x for i, x in enumerate(xs))
        return (2 * weighted) / (n * total) - (n + 1) / n


def load_coverage(spec: RunSpec) -> Coverage:
    """Replay a run's trace and accumulate training-example coverage.

    One "slot" is one (proposal, example) pair: a minibatch of size 3 consumed
    by one proposal contributes 3 slots. Slots — not iterations — are the honest
    x-axis, because the two strategies differ in how many iterations they fit
    into the same rollout budget.
    """
    trace_path = spec.run_dir / "run_log.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"{trace_path} not found; run_dir has no persisted trace")
    trace = json.loads(trace_path.read_text())

    cov = Coverage(spec=spec)
    seen: set[int] = set()
    slots = 0
    for entry in trace:
        tasks = entry.get("tasks")
        if not tasks:
            # Pre-#329 traces only kept the first task's ids at the top level.
            ids = entry.get("subsample_ids")
            tasks = [{"subsample_ids": ids}] if ids else []

        # Mask dead iterations: every task sampled a minibatch but none was ever
        # evaluated against a child, so the iteration carries no signal about
        # which examples are worth sampling. Counting its visits would inflate
        # the profile and stretch the coverage curve with slots that never had a
        # chance to induce anything.
        live = [x for x in tasks if x.get("new_subsample_scores") is not None]
        if tasks and not live:
            cov.n_dead_iterations += 1
            cov.n_masked_tasks += len(tasks)
            cov.n_masked_slots += sum(len(x.get("subsample_ids") or []) for x in tasks)
            continue

        for task in live:
            ids = task.get("subsample_ids") or []
            for example_id in ids:
                cov.counts[example_id] += 1
                seen.add(example_id)
                slots += 1
            if ids:
                cov.n_proposals += 1
                cov.slots.append(slots)
                cov.distinct.append(len(seen))

            # Did this minibatch induce a new candidate? The proposer records
            # the parent's and child's per-example minibatch scores; the engine
            # accepts on sum(after) > sum(before) (StrictImprovementAcceptance,
            # the default). Verified against ``new_program_indices``: summing
            # these per-task improvements reproduces the accepted-candidate
            # count exactly in every run plotted here.
            # Every surviving task has both score vectors, so it is a real
            # attempt. The engine accepts on sum(after) > sum(before)
            # (StrictImprovementAcceptance, the default).
            before = task["subsample_scores"]
            after = task["new_subsample_scores"]
            for example_id in ids:
                cov.attempts[example_id] += 1
            if sum(after) > sum(before):
                for example_id in ids:
                    cov.induced[example_id] += 1
        cov.n_iterations += 1
    return cov


def _style_axes(ax, t: dict, *, xlabel: str, ylabel: str) -> None:
    """Recessive hairline chrome: solid gridlines one shade off the surface."""
    ax.set_facecolor(t["surface"])
    ax.set_xlabel(xlabel, color=t["ink2"], fontsize=10)
    ax.set_ylabel(ylabel, color=t["ink2"], fontsize=10)
    ax.grid(True, which="major", color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9, length=3, width=0.8)


def plot_dataset(covs: list[Coverage], dataset_label: str, mode: str, out_path: Path) -> None:
    """Two panels for one dataset: coverage curve + visit profile.

    Both panels share one y-meaning per panel and a single axis each — no
    dual-axis. The two strategies are the subject, so colour is categorical
    (2 series), a legend is always present, and endpoints are direct-labeled.
    """
    t = THEME[mode]
    plt.rcParams["font.sans-serif"] = FONTS
    plt.rcParams["axes.unicode_minus"] = False

    # Grow the header band when a run needs a caveat continuation line.
    n_caveats = sum(1 for c in covs if c.n_dead_iterations or c.spec.note)
    header_h = 0.048 * (len(covs) + n_caveats) + 0.085
    fig, (ax_cov, ax_prof) = plt.subplots(1, 2, figsize=(12.5, 5.3 + 0.32 * n_caveats))
    fig.patch.set_facecolor(t["surface"])

    n_train = covs[0].spec.trainset_size

    # --- Panel 1: cumulative distinct examples vs slots consumed -----------
    # Label the slot count at which each run first reaches full coverage — the
    # comparison the panel exists to make. Selective, never one per point.
    for cov in covs:
        color = t[cov.spec.strategy]
        ax_cov.step(
            [0, *cov.slots],
            [0, *cov.distinct],
            where="post",
            color=color,
            linewidth=2.0,
            label=cov.spec.label,
            solid_capstyle="round",
        )
        reach = cov.slots_to_full_coverage
        if reach is not None:
            ax_cov.plot(
                [reach], [cov.spec.trainset_size], marker="o", markersize=6,
                color=color, markeredgecolor=t["surface"], markeredgewidth=2, zorder=4,
            )
            ax_cov.annotate(
                f"{reach} slots",
                xy=(reach, cov.spec.trainset_size),
                xytext=(4, -12),
                textcoords="offset points",
                color=color,
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax_cov.annotate(
                f"{cov.n_distinct}/{n_train}",
                xy=(cov.slots[-1], cov.distinct[-1]),
                xytext=(6, -1),
                textcoords="offset points",
                color=color,
                fontsize=9,
                fontweight="bold",
                va="center",
            )

    ax_cov.axhline(n_train, color=t["axis"], linewidth=1.0, linestyle="-")
    ax_cov.annotate(
        f"full trainset ({n_train})",
        xy=(0, n_train),
        xytext=(2, 4),
        textcoords="offset points",
        color=t["muted"],
        fontsize=8.5,
    )
    # Once every run has saturated, the rest of the curve is a flat line that
    # just compresses the informative part. Clip shortly past the last run to
    # reach full coverage (keeping the full range when someone never gets there).
    reaches = [c.slots_to_full_coverage for c in covs]
    run_end = max(c.slots[-1] for c in covs)
    if all(r is not None for r in reaches):
        x_max = min(run_end, int(max(r for r in reaches if r is not None) * 1.18))
    else:
        x_max = run_end
    clipped = x_max < run_end

    _style_axes(
        ax_cov,
        t,
        xlabel="minibatch slots consumed  (proposals x minibatch size)"
        + ("  - axis clipped past full coverage" if clipped else ""),
        ylabel="distinct training examples seen",
    )
    ax_cov.set_ylim(0, n_train * 1.12)
    ax_cov.set_xlim(0, x_max * 1.02)
    ax_cov.set_title("Coverage speed", color=t["ink"], fontsize=11, fontweight="bold", loc="left", pad=8)
    leg = ax_cov.legend(
        loc="lower right", frameon=False, fontsize=9.5, labelcolor=t["ink2"], handlelength=1.6
    )
    for text in leg.get_texts():
        text.set_color(t["ink2"])

    # --- Panel 2: visits (step outline) + inductions (solid bars) ----------
    # Both series count the SAME unit — minibatch slots for one example — and
    # inductions are a subset of visits, so the bars nest inside the step on
    # ONE shared axis. No second y-scale.
    # Ranks are positional: the same x is NOT the same example across strategies.
    y_top = max(max(c.visit_profile) for c in covs)
    n_series = len(covs)
    bar_w = 0.78 / n_series
    for k, cov in enumerate(covs):
        color = t[cov.spec.strategy]
        profile = cov.visit_profile
        xs = [*range(n_train), n_train]
        ys = [*profile, profile[-1]]
        ax_prof.fill_between(xs, ys, step="post", color=color, alpha=0.10, linewidth=0)
        ax_prof.step(
            xs, ys, where="post", color=color, linewidth=2.0, label=f"{cov.spec.label} - sampled"
        )

        induced = cov.induced_profile()
        # Offset the two strategies' bars within each rank; the gap between
        # them is surface, not a border.
        offsets = [r + 0.11 + bar_w * (k + 0.5) for r in range(n_train)]
        ax_prof.bar(
            offsets,
            induced,
            width=bar_w * 0.80,
            color=color,
            linewidth=0,
            zorder=3,
            label=f"{cov.spec.label} - induced a candidate",
        )

    _style_axes(
        ax_prof,
        t,
        xlabel="training examples, ranked by how often sampled",
        ylabel="minibatch slots  (sampled / of which induced)",
    )
    ax_prof.set_xlim(0, n_train)
    ax_prof.set_ylim(0, y_top * 1.30)
    ax_prof.grid(False, axis="x")
    ax_prof.set_title(
        "Visits vs. successful inductions",
        color=t["ink"],
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    leg2 = ax_prof.legend(
        loc="upper right", frameon=False, fontsize=8.5, labelcolor=t["ink2"], handlelength=1.6
    )
    for text in leg2.get_texts():
        text.set_color(t["ink2"])

    fig.suptitle(
        f"Training-set coverage - {dataset_label}",
        color=t["ink"],
        fontsize=13,
        fontweight="bold",
        x=0.008,
        ha="left",
        y=0.995,
        va="top",
    )
    # One header line per strategy, in its own hue, carrying the summary stats
    # that would otherwise collide with the panel titles.
    # Interleave each run's stat line with its caveat continuation line, so a
    # run with caveats pushes the next run's line down instead of overlapping it.
    y = 0.928
    for cov in covs:
        reach = cov.slots_to_full_coverage
        reach_s = f"full coverage at {reach} slots" if reach is not None else "never fully covered"
        line = (
            f"{cov.spec.label}: {cov.n_iterations} live iters, {cov.total_slots} slots, "
            f"Gini {cov.gini:.2f}  |  {reach_s}  |  induced {cov.total_induced}/"
            f"{cov.total_attempts} ({cov.induction_rate:.0%}), "
            f"{cov.n_induced_examples}/{n_train} examples ever induced"
        )
        fig.text(0.008, y, line, color=t[cov.spec.strategy], fontsize=8.5, ha="left")
        y -= 0.036
        # Caveats go on their own continuation line so nothing runs off the edge.
        caveats = []
        if cov.n_dead_iterations:
            caveats.append(
                f"masked {cov.n_dead_iterations} dead iters ({cov.n_masked_slots} slots, "
                "reflection LM out of credit: budget spent, no candidate produced)"
            )
        if cov.spec.note:
            caveats.append(cov.spec.note)
        if caveats:
            fig.text(0.020, y, "^ " + "; ".join(caveats), color=t["muted"], fontsize=8, ha="left")
            y -= 0.036

    fig.tight_layout(rect=(0, 0, 1, 1.0 - header_h))
    fig.savefig(out_path, dpi=170, facecolor=t["surface"])
    plt.close(fig)


def write_table(covs: list[tuple[str, Coverage]], out_path: Path) -> None:
    """Table view: every plotted value reachable without reading the figure."""
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "run_label",
                "strategy",
                "run_dir",
                "budget_metric_calls",
                "live_iterations",
                "proposals",
                "trainset_size",
                "minibatch_slots",
                "distinct_examples",
                "coverage_frac",
                "slots_to_full_coverage",
                "max_visits",
                "min_visits",
                "mean_visits",
                "gini",
                "evaluated_attempt_slots",
                "induced_slots",
                "induction_rate",
                "examples_ever_induced",
                "max_inductions",
                "masked_dead_iterations",
                "masked_slots",
                "note",
            ]
        )
        for dataset, cov in covs:
            profile = cov.visit_profile
            w.writerow(
                [
                    dataset,
                    cov.spec.label,
                    cov.spec.strategy,
                    cov.spec.run_dir.relative_to(REPO_ROOT).as_posix(),
                    cov.spec.budget,
                    cov.n_iterations,
                    cov.n_proposals,
                    cov.spec.trainset_size,
                    cov.total_slots,
                    cov.n_distinct,
                    f"{cov.coverage_frac:.4f}",
                    cov.slots_to_full_coverage if cov.slots_to_full_coverage is not None else "",
                    max(profile),
                    min(profile),
                    f"{cov.total_slots / cov.spec.trainset_size:.3f}",
                    f"{cov.gini:.4f}",
                    cov.total_attempts,
                    cov.total_induced,
                    f"{cov.induction_rate:.4f}",
                    cov.n_induced_examples,
                    max(cov.induced_profile()),
                    cov.n_dead_iterations,
                    cov.n_masked_slots,
                    cov.spec.note,
                ]
            )


# --- Run registry -----------------------------------------------------------
# trainset_size / budget come from the launch scripts in scripts/formal/:
#   math500: AIME_TRAIN_K=40, budget 1000 (run_math500_gepa_np.sh /
#            run_math500_gepa_captransfer.sh); the 500-budget baseline
#            (run_math500_gepa.sh) is reported in the CSV but not plotted, so
#            the figure compares equal budgets.
#   hmmt:    full split (30), budget 1000 for both.
DATASETS: dict[str, dict] = {
    "math500": {
        "label": "MATH-500  (trainset 40, budget 1000 rollouts)",
        # Output stem, matching results/val_score_vs_budget/'s convention.
        "stem": "math500_captransfer3_vs_baseline",
        "plot": [
            RunSpec(
                label="baseline (stock GEPA)",
                run_dir=TEST_ROOT / "math500_formal/gepa_baseline/t_0.2_run_seed_42_long",
                strategy="baseline",
                trainset_size=40,
                budget=1000,
            ),
            RunSpec(
                label="capability transfer",
                run_dir=TEST_ROOT / "math500_formal/gepa_captransfer_3/t_0.2/run_seed_42",
                strategy="captransfer",
                trainset_size=40,
                budget=1000,
            ),
        ],
        "table_only": [
            RunSpec(
                label="baseline (500-rollout budget)",
                run_dir=TEST_ROOT / "math500_formal/gepa_baseline/t_0.2_run_seed_42",
                strategy="baseline",
                trainset_size=40,
                budget=500,
                note="half budget; excluded from the figure",
            ),
        ],
    },
    "hmmt": {
        "label": "HMMT  (trainset 30, dead iterations masked)",
        "stem": "hmmt_captransfer_vs_baseline",
        "plot": [
            RunSpec(
                label="baseline (stock GEPA)",
                run_dir=TEST_ROOT / "hmmt_formal/gepa_baseline/t_0.2/run_seed_42",
                strategy="baseline",
                trainset_size=30,
                budget=1188,
                note="budget topped up after the credit outage",
            ),
            RunSpec(
                label="capability transfer",
                run_dir=TEST_ROOT / "hmmt_formal/gepa_captransfer/t_0.2/run_seed_42",
                strategy="captransfer",
                trainset_size=30,
                budget=1023,
            ),
        ],
        # The second baseline attempt stalled on the same credit outage at i8 and
        # never recovered, so only 8 of its 13 iterations are live.
        "table_only": [
            RunSpec(
                label="baseline attempt 2",
                run_dir=TEST_ROOT / "hmmt_formal/gepa_baseline2/t_0.2/run_seed_42",
                strategy="baseline",
                trainset_size=30,
                budget=678,
                note="stalled at i8 on the credit outage; excluded from the figure",
            ),
        ],
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "results" / "train_coverage"),
        help="where to write figures (one subfolder, like results/val_score_vs_budget/)",
    )
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    table_rows: list[tuple[str, Coverage]] = []
    for name, cfg in DATASETS.items():
        covs = [load_coverage(s) for s in cfg["plot"]]
        stem = cfg["stem"]
        for mode in ("light", "dark"):
            suffix = "" if mode == "light" else "_dark"
            out = outdir / f"{stem}{suffix}.png"
            plot_dataset(covs, cfg["label"], mode, out)
            print(f"wrote {out}")

        rows = [(name, c) for c in covs] + [(name, load_coverage(s)) for s in cfg["table_only"]]
        csv_path = outdir / f"{stem}.csv"
        write_table(rows, csv_path)
        print(f"wrote {csv_path}")
        table_rows += rows

    for name, cov in table_rows:
        reach = cov.slots_to_full_coverage
        reach_s = f"{reach} slots" if reach is not None else "never"
        extra = (
            f"  [masked {cov.n_dead_iterations} dead iters / {cov.n_masked_slots} slots]"
            if cov.n_dead_iterations
            else ""
        )
        print(
            f"{name:8s} {cov.spec.label:30s} "
            f"cov {cov.n_distinct:3d}/{cov.spec.trainset_size} ({cov.coverage_frac:6.1%})  "
            f"full at {reach_s:>11s}  Gini {cov.gini:.3f}  slots {cov.total_slots}  "
            f"induced {cov.total_induced:3d}/{cov.total_attempts:3d} ({cov.induction_rate:5.1%}), "
            f"{cov.n_induced_examples}/{cov.spec.trainset_size} examples{extra}"
        )


if __name__ == "__main__":
    main()
