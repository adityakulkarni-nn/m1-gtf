"""
=============================================================================
master_error_analysis.py
=============================================================================
Full statistical analysis and visualisation of targeting error data stored in
master_error.xlsx, which contains two experimental sheets:

  1. adjusted_position  – errors measured on the standard adjusted-position
                          phantom setup; operators are labelled MAH / CO / AK.
  2. adapter            – errors measured with the adapter attachment; operator
                          column stores integer codes (1 / 2 / 3).
                          Rows where both `collar` AND `arc` are zero / NaN are
                          excluded (those targets had no valid adapter setup).

For each sheet the three operators' readings are averaged per target to obtain
one mean error value per target per sheet.  The four error metrics compared are:
    error_x, error_y, error_z, total_error

Figures produced
----------------
  Fig 1  – 2 × 4 box-plot grid  (adjusted_position | adapter) × (4 error axes)
  Fig 2  – Per-target mean error bar chart, both sheets side-by-side
  Fig 3  – Radar / spider chart of mean error profile per target (both sheets)
  Fig 4  – Heat-maps of raw error values  (targets × operators, one per axis)
  Fig 5  – Correlation matrix for the four error metrics in each sheet
  Fig 6  – Scatter matrix  (pair plot) for adjusted_position
  Fig 7  – Scatter matrix  (pair plot) for adapter
  Fig 8  – Error-axis contribution stacked bar chart per target (both sheets)

All plots are saved as PNG files next to this script.
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy import stats
from itertools import combinations
import textwrap

# ──────────────────────────────────────────────────────────────────────────────
# 0.  PALETTE / STYLE CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
SHEET_COLORS  = {"adjusted_position": "#2E86AB", "adapter": "#E84855"}
METRIC_COLORS = {
    "error_x"    : "#4C72B0",
    "error_y"    : "#DD8452",
    "error_z"    : "#55A868",
    "total_error": "#C44E52",
}
METRICS = ["error_x", "error_y", "error_z", "total_error"]
METRIC_LABELS = {
    "error_x"    : "Error X (mm)",
    "error_y"    : "Error Y (mm)",
    "error_z"    : "Error Z (mm)",
    "total_error": "Total Error (mm)",
}

plt.rcParams.update({
    "figure.dpi"      : 150,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "font.family"     : "DejaVu Sans",
    "axes.labelsize"  : 10,
    "axes.titlesize"  : 11,
    "xtick.labelsize" : 9,
    "ytick.labelsize" : 9,
    "legend.fontsize" : 9,
})

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
FILE = "master_error.xlsx"

df_adj_raw = pd.read_excel(FILE, sheet_name="adjusted_position")
df_ada_raw = pd.read_excel(FILE, sheet_name="adapter")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  CLEANING & FILTERING
# ──────────────────────────────────────────────────────────────────────────────

# --- adjusted_position: all rows are valid (collar & arc always present) ---
df_adj = df_adj_raw.copy()

# Standardise operator column to string so we can group uniformly
df_adj["operator"] = df_adj["operator"].astype(str)

# --- adapter: drop rows where BOTH collar AND arc are zero / NaN ---
# A row is "invalid" when it has no physical setup (collar == 0/NaN AND arc == 0/NaN).
def is_invalid_adapter_row(row):
    collar_bad = pd.isna(row["collar"]) or row["collar"] == 0
    arc_bad    = pd.isna(row["arc"])    or row["arc"]    == 0
    return collar_bad and arc_bad

mask_invalid = df_ada_raw.apply(is_invalid_adapter_row, axis=1)
df_ada = df_ada_raw[~mask_invalid].copy()

# Operator in adapter sheet is an integer code; keep as-is for grouping
df_ada["operator"] = df_ada["operator"].astype(str)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  AVERAGE ACROSS OPERATORS  (per target per sheet)
# ──────────────────────────────────────────────────────────────────────────────
# For each target_id we compute the mean of each error metric across all operators.
# This gives us one representative row per target — useful for direct target comparison.

def mean_per_target(df):
    """Return DataFrame with one row per target_id; mean of error metrics."""
    return (
        df.groupby("target_id")[METRICS]
          .mean()
          .reset_index()
    )

avg_adj = mean_per_target(df_adj)
avg_ada = mean_per_target(df_ada)

# Targets present in BOTH sheets (needed for cross-sheet comparison)
common_targets = sorted(
    set(avg_adj["target_id"]) & set(avg_ada["target_id"])
)
print(f"Targets in adjusted_position : {sorted(avg_adj['target_id'].tolist())}")
print(f"Targets in adapter (filtered): {sorted(avg_ada['target_id'].tolist())}")
print(f"Common targets               : {common_targets}")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  DESCRIPTIVE STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def describe_sheet(df, name):
    """Print full descriptive statistics for a given sheet DataFrame."""
    print(f"\n{'='*60}")
    print(f" DESCRIPTIVE STATISTICS — {name.upper()}")
    print(f"{'='*60}")
    desc = df[METRICS].describe().T
    desc["cv_%"] = (desc["std"] / desc["mean"] * 100).round(2)   # coefficient of variation
    print(desc.to_string())

describe_sheet(df_adj, "adjusted_position (all rows)")
describe_sheet(df_ada, "adapter (filtered rows)")

# Per-target summary — useful for spotting problematic targets
print("\n--- Per-target means: adjusted_position ---")
print(avg_adj.to_string(index=False))
print("\n--- Per-target means: adapter ---")
print(avg_ada.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# 5.  STATISTICAL SIGNIFICANCE  (paired t-test on common targets)
# ──────────────────────────────────────────────────────────────────────────────
# For each error metric we test H0: mean(adjusted_position) == mean(adapter)
# using a paired t-test on the common targets' per-target averages.
# Significance threshold α = 0.05.

adj_common = avg_adj[avg_adj["target_id"].isin(common_targets)].set_index("target_id")
ada_common = avg_ada[avg_ada["target_id"].isin(common_targets)].set_index("target_id")

print(f"\n{'='*60}")
print(" PAIRED T-TEST: adjusted_position vs. adapter (common targets)")
print(f"{'='*60}")
for m in METRICS:
    t, p = stats.ttest_rel(adj_common[m], ada_common[m])
    sig = "*** SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"  {m:<15}  t = {t:+.3f}   p = {p:.4f}   → {sig}")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  CORRELATIONS
# ──────────────────────────────────────────────────────────────────────────────
corr_adj = df_adj[METRICS].corr()
corr_ada = df_ada[METRICS].corr()

print(f"\n{'='*60}")
print(" PEARSON CORRELATION MATRIX — adjusted_position")
print(f"{'='*60}")
print(corr_adj.round(3).to_string())

print(f"\n{'='*60}")
print(" PEARSON CORRELATION MATRIX — adapter")
print(f"{'='*60}")
print(corr_ada.round(3).to_string())

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: add_stats_annotation
# Annotates a box plot axis with min / median / max text labels.
# ──────────────────────────────────────────────────────────────────────────────

def annotate_box(ax, data_list, positions, color="dimgray"):
    """Overlay min / median / max on each box."""
    for pos, data in zip(positions, data_list):
        data = np.array(data, dtype=float)
        data = data[~np.isnan(data)]
        if len(data) == 0:
            continue
        med = np.median(data)
        ax.text(pos, med, f"{med:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=color,
                fontweight="bold")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — 2 × 4 BOX-PLOT GRID
# Row 0 = adjusted_position   |   Row 1 = adapter
# Columns = error_x, error_y, error_z, total_error
# Each box plot groups data by target_id so we can see per-target spread.
# ──────────────────────────────────────────────────────────────────────────────

def make_boxplot_row(axes_row, df, sheet_name, targets):
    """Fill one row of the 2×4 grid with box plots + jittered dots (one per error metric)."""
    color      = SHEET_COLORS[sheet_name]
    dot_color  = "#1a1a2e" if sheet_name == "adjusted_position" else "#7a0010"
    rng        = np.random.default_rng(42)   # reproducible jitter

    for ax, metric in zip(axes_row, METRICS):
        groups = [df.loc[df["target_id"] == t, metric].dropna().values
                  for t in targets]

        # ── box plot ──────────────────────────────────────────────────────
        bp = ax.boxplot(
            groups,
            patch_artist=True,
            widths=0.45,
            whis=(0, 100),                             # whiskers span full data range
            showfliers=False,                          # fliers shown as dots instead
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color=color, linewidth=1.4, linestyle="--"),
            capprops=dict(color=color, linewidth=1.8),
            boxprops=dict(linewidth=1.4),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        # ── jittered data dots ────────────────────────────────────────────
        for pos, data in zip(range(1, len(targets) + 1), groups):
            if len(data) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(data))
            ax.scatter(
                pos + jitter, data,
                color=dot_color, s=45, alpha=0.78,
                zorder=3, linewidths=0.5,
                edgecolors="white"
            )

        # ── median annotation ─────────────────────────────────────────────
        annotate_box(ax, groups, range(1, len(targets) + 1))

        # ── axes styling ──────────────────────────────────────────────────
        ax.set_xticks(range(1, len(targets) + 1))
        ax.set_xticklabels(targets, rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(
            f"{METRIC_LABELS[metric]}",
            fontsize=12, fontweight="bold", pad=8
        )
        # ── fine y-axis scale ──────────────────────────────────────────────
        # Tighten the y range to the actual data spread (5 % padding each side)
        all_vals = np.concatenate([g for g in groups if len(g) > 0])
        if len(all_vals) > 0:
            data_min, data_max = all_vals.min(), all_vals.max()
            span = max(data_max - data_min, 0.1)   # guard against flat data
            pad  = span * 0.12
            ax.set_ylim(data_min - pad, data_max + pad * 2.5)

        # Fine major ticks: ~6-8 intervals across the visible range
        ax.yaxis.set_major_locator(AutoMinorLocator(n=1))   # let matplotlib choose majors
        # Force a reasonable number of major ticks
        y_lo, y_hi = ax.get_ylim()
        tick_span = y_hi - y_lo
        # Pick a step size that gives ~6 major ticks
        raw_step  = tick_span / 6
        magnitude = 10 ** np.floor(np.log10(raw_step))
        nice_steps = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0]
        step = magnitude * min(nice_steps, key=lambda s: abs(s - raw_step / magnitude))
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))     # 1 minor between each major

        ax.yaxis.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", which="major", labelsize=9, length=5, width=0.9)
        ax.tick_params(axis="y", which="minor", length=3, width=0.6)
        # n= count above the bottom of the visible axis
        y_lo2, _ = ax.get_ylim()
        for pos, data in zip(range(1, len(targets) + 1), groups):
            ax.text(pos, y_lo2, f"n={len(data)}", ha="center", va="bottom",
                    fontsize=8, color="gray")

# All targets per sheet
targets_adj = [t for t in sorted(df_adj["target_id"].unique()) if t in common_targets]
targets_ada = [t for t in sorted(df_ada["target_id"].unique()) if t in common_targets]

fig1, axes1 = plt.subplots(2, 4, figsize=(36, 18),
                           gridspec_kw={"hspace": 0.45, "wspace": 0.32})
fig1.patch.set_facecolor("#fafafa")
fig1.suptitle(
    "Targeting Error Distribution per Target",
    fontsize=16, fontweight="bold"
)

make_boxplot_row(axes1[0], df_adj, "adjusted_position", targets_adj)
make_boxplot_row(axes1[1], df_ada, "adapter",           targets_ada)

# Add row labels as the leftmost ylabel of the first column subplot
for row_idx, (ax_row, label) in enumerate(zip(
    [axes1[0][0], axes1[1][0]],
    ["Adjusted Position", "Adapter (filtered)"]
)):
    ax_row.set_ylabel(
        f"{label}\n{ax_row.get_ylabel()}",
        fontsize=11, fontweight="bold",
        color=list(SHEET_COLORS.values())[row_idx],
        labelpad=8
    )

# Legend: colour patches + dot markers
patch_adj = mpatches.Patch(color=SHEET_COLORS["adjusted_position"], alpha=0.55,
                            label="adjusted_position (box)")
patch_ada = mpatches.Patch(color=SHEET_COLORS["adapter"], alpha=0.55,
                            label="adapter (box, filtered)")
dot_adj   = mlines.Line2D([], [], color="#1a1a2e", marker="o", linestyle="None",
                           markersize=6, label="adjusted_position (observations)")
dot_ada   = mlines.Line2D([], [], color="#7a0010", marker="o", linestyle="None",
                           markersize=6, label="adapter (observations)")
fig1.legend(
    handles=[patch_adj, dot_adj, patch_ada, dot_ada],
    loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
    frameon=True, framealpha=0.9, edgecolor="#cccccc", fontsize=10
)

fig1.tight_layout(rect=[0, 0.04, 1, 0.97])
fig1.savefig("fig1_boxplots_2x4.png", bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close(fig1)
print("\nSaved: fig1_boxplots_2x4.png")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — PER-TARGET MEAN ERROR BAR CHART
# Both sheets plotted side-by-side for each metric, only common targets.
# Gives an immediate visual of which targets perform better/worse under each
# experimental condition.
# ──────────────────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(2, 2, figsize=(26, 16))
fig2.suptitle(
    "Per-Target Mean Error: adjusted_position vs. adapter\n(common targets, averaged across operators)",
    fontsize=13, fontweight="bold"
)

x = np.arange(len(common_targets))
w = 0.38  # bar width

for ax, metric in zip(axes2.flatten(), METRICS):
    vals_adj = [adj_common.loc[t, metric] for t in common_targets]
    vals_ada = [ada_common.loc[t, metric] for t in common_targets]

    bars1 = ax.bar(x - w/2, vals_adj, width=w, color=SHEET_COLORS["adjusted_position"],
                   alpha=0.82, label="adjusted_position")
    bars2 = ax.bar(x + w/2, vals_ada, width=w, color=SHEET_COLORS["adapter"],
                   alpha=0.82, label="adapter")

    # Value labels on top of bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(common_targets)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric], fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

fig2.tight_layout()
fig2.savefig("fig2_per_target_bar.png", bbox_inches="tight")
plt.close(fig2)
print("Saved: fig2_per_target_bar.png")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — RADAR / SPIDER CHART OF MEAN ERROR PROFILE PER TARGET
# Shows the "shape" of errors (x, y, z, total) for each target.
# One radar per target; left column = adjusted_position, right = adapter.
# ──────────────────────────────────────────────────────────────────────────────

radar_metrics = ["error_x", "error_y", "error_z", "total_error"]
angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]           # close the polygon
labels_radar = ["Error X", "Error Y", "Error Z", "Total\nError"]

def draw_radar(ax, values, color, label):
    vals = values + values[:1]
    ax.plot(angles, vals, color=color, linewidth=2, label=label)
    ax.fill(angles, vals, color=color, alpha=0.20)

n_targets = len(common_targets)
fig3, axes3 = plt.subplots(
    2, n_targets,
    figsize=(10 * n_targets, 22),
    subplot_kw={"polar": True}
)
fig3.suptitle(
    "Radar Error Profiles per Target (common targets)\n"
    "Top: adjusted_position  |  Bottom: adapter",
    fontsize=13, fontweight="bold"
)

for j, (df_src, sheet) in enumerate([
    (avg_adj, "adjusted_position"),
    (avg_ada, "adapter"),
]):
    for i, t in enumerate(common_targets):
        ax = axes3[j][i]
        row = df_src[df_src["target_id"] == t]
        if row.empty:
            ax.set_visible(False)
            continue
        vals = row[radar_metrics].values.flatten().tolist()
        color = SHEET_COLORS[sheet]
        draw_radar(ax, vals, color, sheet)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_radar, fontsize=9)
        ax.set_title(f"{t}\n{sheet}", fontsize=9, pad=12)
        max_val = max(vals) * 1.2 or 1
        ax.set_ylim(0, max_val)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig("fig3_radar_per_target.png", bbox_inches="tight")
plt.close(fig3)
print("Saved: fig3_radar_per_target.png")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — HEAT-MAPS (targets × operators) FOR EACH METRIC
# Two panels per metric (adjusted_position and adapter).
# Intensity = error value; darker = higher error.
# Helps spot if a specific operator consistently under- or over-shoots.
# ──────────────────────────────────────────────────────────────────────────────

cmap_custom = LinearSegmentedColormap.from_list(
    "err_cmap", ["#d0f0c0", "#f6d860", "#e84855"]
)

fig4, axes4 = plt.subplots(2, len(METRICS), figsize=(28, 16))
fig4.suptitle(
    "Heat-Maps of Error by Target × Operator\nTop: adjusted_position  |  Bottom: adapter",
    fontsize=13, fontweight="bold"
)

def make_heatmap_matrix(df, metric):
    """Return pivot matrix (targets as rows, operators as columns)."""
    return df.pivot_table(index="target_id", columns="operator",
                          values=metric, aggfunc="mean")

for col_idx, metric in enumerate(METRICS):
    for row_idx, (df_src, sheet) in enumerate([
        (df_adj, "adjusted_position"),
        (df_ada, "adapter"),
    ]):
        ax = axes4[row_idx][col_idx]
        mat = make_heatmap_matrix(df_src, metric)
        im = ax.imshow(mat.values, cmap=cmap_custom, aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"Op {c}" for c in mat.columns], fontsize=9)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=9)
        ax.set_title(f"{METRIC_LABELS[metric]}\n({sheet})", fontsize=10, fontweight="bold")

        # Annotate each cell with its value
        for (i_r, i_c), val in np.ndenumerate(mat.values):
            if not np.isnan(val):
                ax.text(i_c, i_r, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

fig4.tight_layout()
fig4.savefig("fig4_heatmaps.png", bbox_inches="tight")
plt.close(fig4)
print("Saved: fig4_heatmaps.png")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — CORRELATION MATRICES
# Visualises Pearson correlations among error_x, error_y, error_z, total_error.
# Strong correlation between component errors and total_error is expected;
# cross-axis correlations reveal systematic bias patterns.
# ──────────────────────────────────────────────────────────────────────────────

short_labels = ["X", "Y", "Z", "Total"]
cmap_corr = LinearSegmentedColormap.from_list(
    "corr_cmap", ["#2E86AB", "#FFFFFF", "#E84855"]
)

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(22, 10))
fig5.suptitle("Pearson Correlation Matrix of Error Metrics",
              fontsize=13, fontweight="bold")

def draw_corr_heatmap(ax, corr_df, title):
    mat = corr_df.values
    im  = ax.imshow(mat, cmap=cmap_corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(short_labels)));  ax.set_xticklabels(short_labels)
    ax.set_yticks(range(len(short_labels)));  ax.set_yticklabels(short_labels)
    ax.set_title(title, fontweight="bold")
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=11, color="black" if abs(val) < 0.7 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)

draw_corr_heatmap(ax5a, corr_adj, "adjusted_position")
draw_corr_heatmap(ax5b, corr_ada, "adapter")

fig5.tight_layout()
fig5.savefig("fig5_correlations.png", bbox_inches="tight")
plt.close(fig5)
print("Saved: fig5_correlations.png")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — STACKED BAR: ERROR AXIS CONTRIBUTIONS PER TARGET
# Stacks error_x, error_y, error_z for each target.
# Side-by-side: adjusted_position vs. adapter (common targets only).
# Reveals which spatial axis contributes most to overall error and whether
# the contribution pattern changes between the two experimental setups.
# ──────────────────────────────────────────────────────────────────────────────

fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(26, 14), sharey=True)
fig8.suptitle(
    "Stacked Error Contributions per Target (averaged across operators)\n"
    "Left: adjusted_position  |  Right: adapter",
    fontsize=13, fontweight="bold"
)

component_metrics = ["error_x", "error_y", "error_z"]
component_colors  = [METRIC_COLORS[m] for m in component_metrics]
x8 = np.arange(len(common_targets))

def stacked_bar(ax, df_avg, title):
    bottom = np.zeros(len(common_targets))
    for k, (m, c) in enumerate(zip(component_metrics, component_colors)):
        vals = np.array([df_avg[df_avg["target_id"] == t][m].values[0]
                         for t in common_targets])
        bars = ax.bar(x8, vals, bottom=bottom, color=c, alpha=0.85, label=METRIC_LABELS[m])
        # Annotate segment values
        for xp, v, b in zip(x8, vals, bottom):
            if v > 0.05:
                ax.text(xp, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x8)
    ax.set_xticklabels(common_targets)
    ax.set_ylabel("Error (mm)")
    ax.set_title(title, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right")

stacked_bar(ax8a, avg_adj[avg_adj["target_id"].isin(common_targets)], "adjusted_position")
stacked_bar(ax8b, avg_ada[avg_ada["target_id"].isin(common_targets)], "adapter (filtered)")

fig8.tight_layout()
fig8.savefig("fig8_stacked_contributions.png", bbox_inches="tight")
plt.close(fig8)
print("Saved: fig8_stacked_contributions.png")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — OVERALL ERROR DISTRIBUTION: 2 × 1 BOX PLOTS
# Top: adjusted_position  |  Bottom: adapter
# Each subplot shows one box per error metric pooled across all targets,
# giving a compact single-panel summary of the four error axes per setup.
# ──────────────────────────────────────────────────────────────────────────────

fig9, axes9 = plt.subplots(2, 1, figsize=(20, 24),
                            gridspec_kw={"hspace": 0.38})
fig9.patch.set_facecolor("#fafafa")
fig9.suptitle(
    "Overall Error Distribution by Metric\n(all targets pooled)",
    fontsize=15, fontweight="bold"
)

rng9 = np.random.default_rng(42)

for ax, (df_src, sheet_name) in zip(axes9, [
    (df_adj, "adjusted_position"),
    (df_ada, "adapter"),
]):
    color     = SHEET_COLORS[sheet_name]
    dot_color = "#1a1a2e" if sheet_name == "adjusted_position" else "#7a0010"

    groups = [df_src[df_src["target_id"].isin(common_targets)][m].dropna().values for m in METRICS]

    bp = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.45,
        whis=(0, 100),
        showfliers=False,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color=color, linewidth=1.4, linestyle="--"),
        capprops=dict(color=color, linewidth=1.8),
        boxprops=dict(linewidth=1.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # Jittered dots
    for pos, data in zip(range(1, len(METRICS) + 1), groups):
        jitter = rng9.uniform(-0.18, 0.18, size=len(data))
        ax.scatter(pos + jitter, data, color=dot_color, s=45, alpha=0.78,
                   zorder=3, linewidths=0.5, edgecolors="white")

    # Median annotations
    annotate_box(ax, groups, range(1, len(METRICS) + 1))

    ax.set_xticks(range(1, len(METRICS) + 1))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=12)
    ax.set_ylabel("Error (mm)", fontsize=12)
    ax.set_title(sheet_name, fontsize=13, fontweight="bold", color=color, pad=8)

    # Fine y-axis scale (same logic as fig1)
    all_vals = np.concatenate([g for g in groups if len(g) > 0])
    data_min, data_max = all_vals.min(), all_vals.max()
    span = max(data_max - data_min, 0.1)
    pad  = span * 0.12
    ax.set_ylim(data_min - pad, data_max + pad * 2.5)

    y_lo, y_hi = ax.get_ylim()
    raw_step   = (y_hi - y_lo) / 6
    magnitude  = 10 ** np.floor(np.log10(raw_step))
    nice_steps = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0]
    step = magnitude * min(nice_steps, key=lambda s: abs(s - raw_step / magnitude))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", which="major", labelsize=10, length=5, width=0.9)
    ax.tick_params(axis="y", which="minor", length=3, width=0.6)

    y_lo2, _ = ax.get_ylim()
    for pos, data in zip(range(1, len(METRICS) + 1), groups):
        ax.text(pos, y_lo2, f"n={len(data)}", ha="center", va="bottom",
                fontsize=9, color="gray")

fig9.tight_layout(rect=[0, 0.01, 1, 0.96])
fig9.savefig("fig9_overall_boxplots_2x1.png", bbox_inches="tight", facecolor=fig9.get_facecolor())
plt.close(fig9)
print("Saved: fig9_overall_boxplots_2x1.png")

# ──────────────────────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────────────────────
print("\n✔  All figures saved. Analysis complete.")
