"""Plot the capability-transfer estimator matrix.

The figure answers: "what does our estimator think each TRAIN example is worth
to each CANDIDATE, and did that belief actually drive sampling?"

The plotted matrix is the estimator's candidate-specific term:

  relevance(k | c) = mean over the validation questions c fails of mu[j,k]
      "which training example teaches what THIS candidate is still missing".
      Most of its variance is within-column, i.e. it genuinely re-ranks
      training examples per candidate.

What actually feeds selection is value(k | c) = U[k]^gamma * relevance(k | c).
U[k] is a column constant, and it dominates: 80-91% of value's variance is
between-column versus 33-44% for relevance alone, so the gate flattens most of
the per-candidate structure shown here. That is why U[k] is drawn as a strip
under the matrix, and why both variance shares stay in the CSV — the figure
shows the belief, the strip shows what damps it.

Cells are ROW-CENTERED: each candidate's row has its own mean subtracted, so
color reads "above/below what this candidate finds averagely useful" on a
diverging scale with warm = more useful (the intuitive direction). Centering is
what makes the per-candidate signal
visible at all — the raw rows carry a large offset driven by n_gaps (a candidate
failing 27 val questions has a uniformly higher mean than one failing 10), and
that offset otherwise swamps the differences in row SHAPE, which is the part
that actually steers sampling. Row means are printed in the row labels and kept
in the CSV, so the removed term stays inspectable.

  ring    train example actually selected into a minibatch for that candidate
          (as parent) by the sampler
  rows    candidates, in natural candidate-index order
  columns train example ids, in natural id order
  strips  U[k] usability gate and n_k pull count, column-aligned

Estimator state is not persisted; it comes from replay_estimator.py, whose
minibatch reconstruction matches the run log exactly (45/45 and 65/65).

Outputs figures/estimator_<dataset>_<mode>.png.
"""

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"

# Output stem per dataset, matching results/train_coverage/'s convention.
PANELS = [
    ("math500_captransfer_3", "math500_captransfer3_estimator"),
    ("hmmt_captransfer", "hmmt_captransfer_estimator"),
]

# Diverging ramp for the row-centered cells: blue (below this candidate's mean)
# <-> neutral gray midpoint (at the mean) <-> orange (above). Warm = the more
# useful end, which is the intuitive reading. Warm/cool poles validate at CVD
# dE 24.7 light / 26.8 dark; the midpoint is gray, never a hue.
DIVERGING_LIGHT = ["#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4", "#f0efec",
                   "#f5bda0", "#f0956b", "#eb6834"]
DIVERGING_DARK = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#383835",
                  "#eaa87e", "#e2814f", "#d95926"]

# Sequential blue for the U[k] strip (a magnitude, not a deviation).
SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_DARK = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

LIGHT = dict(surface="#fcfcfb", plane="#f9f9f7", ink="#0b0b0b", ink2="#52514e",
             muted="#898781", grid="#e1e0d9", axis="#c3c2b7", accent="#0b0b0b",
             border="rgba(11,11,11,0.10)", div=DIVERGING_LIGHT, seq=SEQ_LIGHT)
DARK = dict(surface="#1a1a19", plane="#0d0d0d", ink="#ffffff", ink2="#c3c2b7",
            muted="#898781", grid="#2c2c2a", axis="#383835", accent="#ffffff",
            border="rgba(255,255,255,0.10)", div=DIVERGING_DARK, seq=SEQ_DARK)


def _var_share_between_columns(mat: list[list[float]]) -> float:
    """Fraction of total variance explained by column means (vs. per-candidate)."""
    n_r = len(mat)
    flat = [v for row in mat for v in row]
    gm = sum(flat) / len(flat)
    tot = sum((v - gm) ** 2 for v in flat)
    if tot == 0:
        return 0.0
    n_c = len(mat[0])
    col_means = [sum(mat[r][c] for r in range(n_r)) / n_r for c in range(n_c)]
    betw = n_r * sum((cm - gm) ** 2 for cm in col_means)
    return betw / tot


def _row_center(mat: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """Subtract each row's own mean. Returns (centered matrix, row means)."""
    means = [sum(row) / len(row) for row in mat]
    return [[v - m for v in row] for row, m in zip(mat, means, strict=True)], means


def load(name: str) -> dict:
    est = json.loads((DATA_DIR / f"{name}.estimator.json").read_text())
    run = json.loads((DATA_DIR / f"{name}.json").read_text())

    n_train = est["n_train"]
    train_ids = list(range(n_train))
    by_idx = {c["idx"]: c for c in run["candidates"]}
    gamma = est["ct_kwargs"]["usability_weight"]

    # Rows in natural candidate-index order (0 = seed, then discovery order).
    rows = sorted(int(i) for i in est["candidate_value"])

    # Which train examples were actually selected for each candidate-as-parent.
    selected: dict[int, set[int]] = {i: set() for i in rows}
    for it in est["per_iteration"]:
        for t in it["tasks"]:
            if t["parent_idx"] in selected:
                selected[t["parent_idx"]].update(t["logged"])

    U = {k: est["usability"][str(k)] for k in train_ids}
    gate = {k: (U[k] ** gamma if gamma > 0 else 1.0) for k in train_ids}

    # value is what the estimator emits; relevance is value with the column-only
    # gate divided back out (the candidate-specific term on its own).
    value = {i: {k: est["candidate_value"][str(i)]["value"][str(k)] for k in train_ids} for i in rows}
    relevance = {
        i: {k: (value[i][k] / gate[k] if gate[k] else 0.0) for k in train_ids} for i in rows
    }

    # Columns in natural train-example id order.
    col_order = list(train_ids)

    m_rel = [[relevance[i][k] for k in col_order] for i in rows]
    m_val = [[value[i][k] for k in col_order] for i in rows]
    c_rel, rowmean_rel = _row_center(m_rel)
    c_val, rowmean_val = _row_center(m_val)

    return dict(
        name=name,
        dataset=est["dataset"],
        n_train=n_train,
        n_val=est["n_val"],
        gamma=gamma,
        rows=rows,
        col_order=col_order,
        m_rel=m_rel,
        m_val=m_val,
        c_rel=c_rel,
        c_val=c_val,
        rowmean_rel=rowmean_rel,
        rowmean_val=rowmean_val,
        share_rel=_var_share_between_columns(m_rel),
        share_val=_var_share_between_columns(m_val),
        selected={i: sorted(selected[i]) for i in rows},
        val_avg={i: by_idx[i]["val_avg"] for i in rows},
        n_gaps={i: est["candidate_value"][str(i)]["n_gaps"] for i in rows},
        usability=[U[k] for k in col_order],
        pulls=[est["pulls"][str(k)] for k in col_order],
        fidelity=est["fidelity"],
        n_iterations=run["n_iterations"],
        total_metric_calls=run["total_metric_calls"],
        ct=est["ct_kwargs"],
    )


def write_table(p: dict, path: Path) -> None:
    """Per-column estimator state, so the figure's claims are checkable as numbers."""
    n_r = len(p["rows"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "run_dir_label", "train_id",
            "usability_U", "pulls_n_k",
            "mean_relevance", "min_relevance", "max_relevance",
            "mean_value", "min_value", "max_value",
            # Row-centered relevance = what the figure actually paints.
            "mean_centered_relevance", "min_centered_relevance", "max_centered_relevance",
            "n_candidates_rating_above_own_mean", "n_candidates_selecting",
            "relevance_between_col_var_share", "value_between_col_var_share",
            "gamma", "replay_matches", "replay_predicted",
        ])
        for j, k in enumerate(p["col_order"]):
            rel = [p["m_rel"][r][j] for r in range(n_r)]
            val = [p["m_val"][r][j] for r in range(n_r)]
            cen = [p["c_rel"][r][j] for r in range(n_r)]
            n_sel = sum(1 for c in p["rows"] if k in p["selected"][c])
            w.writerow([
                p["dataset"], p["name"], k,
                f"{p['usability'][j]:.6f}", p["pulls"][j],
                f"{sum(rel) / n_r:.6f}", f"{min(rel):.6f}", f"{max(rel):.6f}",
                f"{sum(val) / n_r:.6f}", f"{min(val):.6f}", f"{max(val):.6f}",
                f"{sum(cen) / n_r:.6f}", f"{min(cen):.6f}", f"{max(cen):.6f}",
                sum(1 for v in cen if v > 0), n_sel,
                f"{p['share_rel']:.6f}", f"{p['share_val']:.6f}",
                p["gamma"], p["fidelity"]["matches"], p["fidelity"]["predicted_minibatches"],
            ])


def render(p: dict, mode: str, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.gridspec import GridSpec

    T = LIGHT if mode == "light" else DARK
    cmap_div = LinearSegmentedColormap.from_list("div_orange_blue", T["div"])
    cmap_seq = LinearSegmentedColormap.from_list("seq_blue", T["seq"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": T["plane"],
        "axes.facecolor": T["surface"],
        "text.color": T["ink"],
        "xtick.color": T["muted"],
        "ytick.color": T["muted"],
    })

    n_r, n_c = len(p["rows"]), p["n_train"]
    row_h = 0.285
    mat_h = n_r * row_h + 0.55
    # relevance matrix, U strip, n_k strip
    heights = [mat_h, 0.30, 0.30, 0.62]
    width = 5.2 + n_c * 0.285
    # Fixed header/footer in INCHES, so the deck's three title lines never
    # collide with panel A's own title regardless of how tall the matrices are.
    head_in, foot_in = 1.86, 0.34
    fig_h = sum(heights) + head_in + foot_in
    fig = plt.figure(figsize=(width, fig_h))
    gs = GridSpec(len(heights), 1, height_ratios=heights, hspace=0.0, figure=fig,
                  left=2.55 / width, right=1 - 1.15 / width,
                  top=1 - head_in / fig_h, bottom=foot_in / fig_h)

    L = 2.55 / width
    fig.text(L, 1 - 0.30 / fig_h,
             f"{p['dataset']} — what our estimator believes each training example is worth",
             fontsize=14.5, fontweight="bold", color=T["ink"], ha="left", va="top")
    fig.text(L, 1 - 0.60 / fig_h,
             f"{n_r} candidates × {n_c} train examples  ·  {p['n_iterations']} iterations, "
             f"{p['total_metric_calls']} rollouts  ·  minibatch replay reproduces the run "
             f"{p['fidelity']['matches']}/{p['fidelity']['predicted_minibatches']}",
             fontsize=9.4, color=T["ink2"], ha="left", va="top")
    ring_word = "black" if mode == "light" else "white"
    fig.text(L, 1 - 0.84 / fig_h,
             f"each row is centered on its own mean (printed as μ) — orange = this candidate rates the "
             f"example above its own average, blue = below;  {ring_word} ring = actually sampled",
             fontsize=9.4, color=T["ink2"], ha="left", va="top")
    fig.text(L, 1 - 1.08 / fig_h,
             f"selection multiplies this by the column-only gate U[k]^γ (γ={p['gamma']:g}), raising "
             f"between-column variance {p['share_rel']:.0%}→{p['share_val']:.0%}",
             fontsize=9.4, color=T["ink2"], ha="left", va="top")

    col_pos = {k: j for j, k in enumerate(p["col_order"])}

    def draw_matrix(ax, mat, rowmeans, title, sub):
        # Symmetric about zero so the gray midpoint sits exactly at each row's
        # mean and equal deviations get equal weight on both poles.
        lim = max(abs(v) for row in mat for v in row)
        norm = Normalize(vmin=-lim, vmax=lim)
        img = ax.imshow(mat, aspect="auto", cmap=cmap_div, norm=norm,
                        interpolation="nearest", origin="upper")
        # 2px surface gap between cells.
        ax.set_xticks([x - 0.5 for x in range(n_c + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(n_r + 1)], minor=True)
        ax.grid(which="minor", color=T["surface"], linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        for r, cand in enumerate(p["rows"]):
            for k in p["selected"][cand]:
                if k in col_pos:
                    ax.add_patch(plt.Circle((col_pos[k], r), 0.29, fill=False,
                                            edgecolor=T["accent"], linewidth=1.5, zorder=5))
        ax.set_yticks(range(n_r))
        ax.set_yticklabels(
            [
                f"#{c}  val {p['val_avg'][c]:.3f}  gaps {p['n_gaps'][c]:>2d}  μ {m:.3f}"
                for c, m in zip(p["rows"], rowmeans, strict=True)
            ],
            fontsize=7.2, color=T["ink2"], fontfamily="DejaVu Sans Mono",
        )
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10.6, color=T["ink"], loc="left", pad=13,
                     fontweight="bold")
        ax.text(0.0, 1.0, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.5, color=T["ink2"])
        return img

    ax1 = fig.add_subplot(gs[0])
    im1 = draw_matrix(
        ax1,
        p["c_rel"],
        p["rowmean_rel"],
        "candidate-specific belief:  relevance(k | c) = mean over c's failed val questions of μ[j,k]",
        f"re-ranks training examples per candidate — {1 - p['share_rel']:.0%} of its variance is "
        f"within-column (candidate-driven), {p['share_rel']:.0%} between-column",
    )

    axu = fig.add_subplot(gs[1], sharex=ax1)
    axu.imshow([p["usability"]], aspect="auto", cmap=cmap_seq,
               norm=Normalize(vmin=0, vmax=max(p["usability"])), interpolation="nearest")
    axu.set_yticks([0])
    axu.set_yticklabels([f"U[k]  {min(p['usability']):.2f}–{max(p['usability']):.2f}"],
                        fontsize=7.2, color=T["muted"], fontfamily="DejaVu Sans Mono")
    axu.set_xticks([])
    for s in axu.spines.values():
        s.set_visible(False)

    axn = fig.add_subplot(gs[2], sharex=ax1)
    axn.bar(range(n_c), p["pulls"], width=0.70, color=T["seq"][3], linewidth=0)
    axn.set_ylim(0, max(p["pulls"]) * 1.30)
    axn.set_yticks([])
    axn.set_xticks(range(n_c))
    axn.set_xticklabels([str(k) for k in p["col_order"]], fontsize=6.8, color=T["muted"])
    axn.set_xlim(-0.5, n_c - 0.5)
    axn.tick_params(length=0, pad=1.5)
    for s in axn.spines.values():
        s.set_visible(False)
    axn.text(-0.004, 0.45, "n_k pulls", transform=axn.transAxes, ha="right", va="center",
             fontsize=7.2, color=T["muted"], fontfamily="DejaVu Sans Mono")
    for j, v in enumerate(p["pulls"]):
        axn.text(j, v, str(v), ha="center", va="bottom", fontsize=6.0, color=T["muted"])
    axn.set_xlabel("train example id", fontsize=8.8, color=T["ink2"], labelpad=4)

    box = ax1.get_position()
    cax = fig.add_axes([box.x1 + 0.012, box.y0 + box.height * 0.22,
                        0.0085, box.height * 0.56])
    cb = fig.colorbar(im1, cax=cax)
    cb.set_label("relevance − row μ", fontsize=8.2, color=T["ink2"])
    cb.ax.tick_params(labelsize=7.0, color=T["muted"], labelcolor=T["muted"], length=2)
    cb.outline.set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=185, facecolor=T["plane"])
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outdir",
        default=str(REPO_ROOT / "results" / "estimator_matrix"),
        help="where to write figures (one subfolder, like results/train_coverage/)",
    )
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, stem in PANELS:
        p = load(name)
        for mode in ("light", "dark"):
            suffix = "" if mode == "light" else "_dark"
            print("wrote", render(p, mode, outdir / f"{stem}{suffix}.png"))
        csv_path = outdir / f"{stem}.csv"
        write_table(p, csv_path)
        print("wrote", csv_path)

    print()
    for name, _stem in PANELS:
        p = load(name)
        print(
            f"{p['dataset']:8s} {len(p['rows']):3d} cands x {p['n_train']:2d} train  "
            f"relevance between-col var {p['share_rel']:6.1%} (candidate-driven "
            f"{1 - p['share_rel']:5.1%})  ->  value between-col {p['share_val']:6.1%} "
            f"(candidate-driven {1 - p['share_val']:5.1%})  "
            f"U {min(p['usability']):.3f}-{max(p['usability']):.3f}  "
            f"n_k {min(p['pulls'])}-{max(p['pulls'])}  "
            f"replay {p['fidelity']['matches']}/{p['fidelity']['predicted_minibatches']}"
        )


if __name__ == "__main__":
    main()
