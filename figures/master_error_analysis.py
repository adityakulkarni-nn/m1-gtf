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

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
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
SIDE_ORDER = {"Left": 0, "Right": 1, "Other / Midline": 2}
SIDE_COLORS = {
    "Left": "#2E86AB",
    "Right": "#F18F01",
    "Other / Midline": "#6C757D",
}
WHITE = "#FFFFFF"
GRID_COLOR = "#D7DEE8"
TEXT_COLOR = "#1F2933"

plt.rcParams.update({
    "figure.dpi"      : 150,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "font.family"     : "DejaVu Sans",
    "font.size"       : 13,
    "axes.labelsize"  : 14,
    "axes.titlesize"  : 15,
    "axes.labelcolor" : TEXT_COLOR,
    "axes.titleweight": "bold",
    "xtick.labelsize" : 12,
    "ytick.labelsize" : 12,
    "xtick.color"     : TEXT_COLOR,
    "ytick.color"     : TEXT_COLOR,
    "legend.fontsize" : 12,
})

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FILE = SCRIPT_DIR / "master_error.xlsx"

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


def normalize_side_label(side):
    text = str(side).strip().upper()
    if text in {"L", "LEFT"}:
        return "Left"
    if text in {"R", "RIGHT"}:
        return "Right"
    if text in {"M", "MIDLINE", "CENTER", "CENTRE"}:
        return "Other / Midline"
    return "Other / Midline"


def resolve_target_side(side_values):
    normalized = side_values.dropna().map(normalize_side_label)
    if normalized.empty:
        return "Other / Midline"

    preferred = normalized[normalized.isin(["Left", "Right"])]
    if not preferred.empty:
        return preferred.mode().iloc[0]
    return normalized.mode().iloc[0]


def build_target_side_map(df):
    if "side" not in df.columns:
        return {}
    return df.groupby("target_id")["side"].apply(resolve_target_side).to_dict()


def get_ordered_targets(targets, target_side_map):
    return sorted(
        list(targets),
        key=lambda target: (
            SIDE_ORDER.get(target_side_map.get(target, "Other / Midline"), 99),
            str(target),
        ),
    )


def get_target_group_spans(targets, target_side_map):
    if not targets:
        return []

    spans = []
    span_start = 0
    current_side = target_side_map.get(targets[0], "Other / Midline")
    for idx, target in enumerate(targets[1:], start=1):
        target_side = target_side_map.get(target, "Other / Midline")
        if target_side != current_side:
            spans.append((span_start, idx - 1, current_side))
            span_start = idx
            current_side = target_side
    spans.append((span_start, len(targets) - 1, current_side))
    return spans


def target_tick_rotation(count):
    if count >= 10:
        return 35, "right"
    return 0, "center"


def nice_step(span):
    span = max(float(span), 0.1)
    raw_step = span / 6
    magnitude = 10 ** np.floor(np.log10(raw_step))
    nice_steps = np.array([0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0])
    normalized = raw_step / magnitude
    return float(magnitude * nice_steps[np.argmin(np.abs(nice_steps - normalized))])


def style_axis(ax):
    ax.set_facecolor(WHITE)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=12, colors=TEXT_COLOR)


def compute_shared_ylim(data_arrays, include_zero=False, top_padding=0.18, bottom_padding=0.10):
    clean_arrays = [np.asarray(arr, dtype=float) for arr in data_arrays if len(arr) > 0]
    if not clean_arrays:
        return (0.0, 1.0)

    all_vals = np.concatenate(clean_arrays)
    all_vals = all_vals[~np.isnan(all_vals)]
    if all_vals.size == 0:
        return (0.0, 1.0)

    data_min = float(all_vals.min())
    data_max = float(all_vals.max())
    if include_zero:
        data_min = min(0.0, data_min)
    span = max(data_max - data_min, 0.1)
    lower = data_min - span * bottom_padding
    upper = data_max + span * top_padding
    if include_zero:
        lower = min(0.0, lower)
    return (lower, upper)


def apply_shared_y_axis(ax, y_limits):
    y_lo, y_hi = y_limits
    if np.isclose(y_lo, y_hi):
        y_hi = y_lo + 1.0
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(MultipleLocator(nice_step(y_hi - y_lo)))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7, color=GRID_COLOR)
    ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35, color=GRID_COLOR)
    ax.set_axisbelow(True)


def heatmap_label_color(value, vmin, vmax):
    if np.isnan(value) or np.isclose(vmax, vmin):
        return "black"
    midpoint = (vmin + vmax) / 2
    return "white" if value >= midpoint else TEXT_COLOR


def save_transposed_overall_boxplots(output_name, y_limits):
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)
    fig.patch.set_facecolor(WHITE)
    fig.suptitle(
        "Overall Error Distribution by Metric\n(all targets pooled, transposed layout)",
        fontsize=17, fontweight="bold"
    )

    rng_local = np.random.default_rng(42)
    for ax, (df_src, sheet_name) in zip(axes, [
        (df_adj, "adjusted_position"),
        (df_ada, "adapter"),
    ]):
        style_axis(ax)
        color = SHEET_COLORS[sheet_name]
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

        for pos, data in zip(range(1, len(METRICS) + 1), groups):
            jitter = rng_local.uniform(-0.18, 0.18, size=len(data))
            ax.scatter(pos + jitter, data, color=dot_color, s=45, alpha=0.78,
                       zorder=3, linewidths=0.5, edgecolors="white")

        annotate_box(ax, groups, range(1, len(METRICS) + 1))
        apply_shared_y_axis(ax, y_limits)
        ax.set_xticks(range(1, len(METRICS) + 1))
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11)
        ax.set_ylabel("Error (mm)", fontsize=12)
        ax.set_title(sheet_name, fontsize=13, fontweight="bold", color=color, pad=8)

        y_lo2, _ = ax.get_ylim()
        for pos, data in zip(range(1, len(METRICS) + 1), groups):
            ax.text(pos, y_lo2, f"n={len(data)}", ha="center", va="bottom", fontsize=9, color="gray")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_name, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved: {output_name}")


target_side_adj = build_target_side_map(df_adj)
target_side_ada = build_target_side_map(df_ada)
combined_target_side = {**target_side_ada, **target_side_adj}

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
common_targets = get_ordered_targets(
    set(avg_adj["target_id"]) & set(avg_ada["target_id"]),
    combined_target_side,
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

def make_boxplot_row(axes_row, df, sheet_name, targets, target_side_map, shared_ylim):
    """Fill one row of the 2×4 grid with box plots + jittered dots (one per error metric)."""
    color      = SHEET_COLORS[sheet_name]
    rng        = np.random.default_rng(42)   # reproducible jitter
    group_spans = get_target_group_spans(targets, target_side_map)
    tick_rotation, tick_align = target_tick_rotation(len(targets))

    for ax, metric in zip(axes_row, METRICS):
        style_axis(ax)
        groups = [df.loc[df["target_id"] == t, metric].dropna().values
                  for t in targets]
        all_vals = np.concatenate([g for g in groups if len(g) > 0]) if any(len(g) > 0 for g in groups) else np.array([])

        for start_idx, end_idx, side_label in group_spans:
            ax.axvspan(start_idx + 0.5, end_idx + 1.5,
                       color=SIDE_COLORS.get(side_label, "#6C757D"),
                       alpha=0.06, zorder=0)

        # ── box plot ──────────────────────────────────────────────────────
        bp = ax.boxplot(
            groups,
            patch_artist=True,
            widths=0.45,
            whis=(0, 100),                             # whiskers span full data range
            showfliers=False,                          # fliers shown as dots instead
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="#111111",
                           markeredgecolor="white", markersize=6),
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color=color, linewidth=1.4, linestyle="--"),
            capprops=dict(color=color, linewidth=1.8),
            boxprops=dict(linewidth=1.4),
        )
        for patch, target in zip(bp["boxes"], targets):
            patch.set_facecolor(SIDE_COLORS.get(target_side_map.get(target, "Other / Midline"), color))
            patch.set_alpha(0.55)

        # ── jittered data dots ────────────────────────────────────────────
        for pos, data, target in zip(range(1, len(targets) + 1), groups, targets):
            if len(data) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(data))
            ax.scatter(
                pos + jitter, data,
                color=SIDE_COLORS.get(target_side_map.get(target, "Other / Midline"), color),
                s=45, alpha=0.78,
                zorder=3, linewidths=0.5,
                edgecolors="white"
            )

        # ── axes styling ──────────────────────────────────────────────────
        ax.set_xticks(range(1, len(targets) + 1))
        ax.set_xticklabels(targets, rotation=tick_rotation, ha=tick_align, fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
        ax.set_title(
            f"{METRIC_LABELS[metric]}",
            fontsize=12, fontweight="bold", pad=8
        )
        apply_shared_y_axis(ax, shared_ylim)
        ax.tick_params(axis="y", which="major", labelsize=9, length=5, width=0.9)
        ax.tick_params(axis="y", which="minor", length=3, width=0.6)
        ax.tick_params(axis="x", labelsize=10, pad=8)
        ax.grid(True, axis="x", linestyle=":", alpha=0.20, color=GRID_COLOR)

        if len(all_vals) > 0:
            overall_mean = float(np.mean(all_vals))
            ax.axhline(
                overall_mean,
                color="#111111",
                linestyle="-.",
                linewidth=1.8,
                alpha=0.85,
                zorder=2,
            )
            mean_label_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                -0.06,
                overall_mean,
                f"{overall_mean:.2f}",
                transform=mean_label_transform,
                ha="right",
                va="center",
                fontsize=8.5,
                color="#111111",
                fontweight="bold",
                clip_on=False,
            )

        y_lo2, y_hi2 = ax.get_ylim()
        y_text = y_hi2 + (y_hi2 - y_lo2) * 0.03
        for start_idx, end_idx, side_label in group_spans:
            midpoint = ((start_idx + 1) + (end_idx + 1)) / 2
            ax.text(
                midpoint, y_text, side_label,
                ha="center", va="bottom", fontsize=10,
                color=SIDE_COLORS.get(side_label, color),
                fontweight="bold", clip_on=False,
            )

# All targets per sheet
targets_adj = [t for t in common_targets if t in set(df_adj["target_id"].unique())]
targets_ada = [t for t in common_targets if t in set(df_ada["target_id"].unique())]

fig1, axes1 = plt.subplots(2, 4, figsize=(36, 18), sharey=True,
                           gridspec_kw={"hspace": 0.45, "wspace": 0.32})
fig1.patch.set_facecolor(WHITE)
fig1.suptitle(
    "Targeting Error Distribution per Target",
    fontsize=17, fontweight="bold"
)

fig1_shared_ylim = compute_shared_ylim(
    [df_adj[m].dropna().to_numpy(dtype=float) for m in METRICS] +
    [df_ada[m].dropna().to_numpy(dtype=float) for m in METRICS]
)

make_boxplot_row(axes1[0], df_adj, "adjusted_position", targets_adj, target_side_adj, fig1_shared_ylim)
make_boxplot_row(axes1[1], df_ada, "adapter",           targets_ada, target_side_ada, fig1_shared_ylim)

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

# Legend: side colours + observations + mean
patch_left = mpatches.Patch(color=SIDE_COLORS["Left"], alpha=0.55,
                            label="Left targets")
patch_right = mpatches.Patch(color=SIDE_COLORS["Right"], alpha=0.55,
                             label="Right targets")
patch_mid = mpatches.Patch(color=SIDE_COLORS["Other / Midline"], alpha=0.55,
                           label="Other / midline")
dot_obs   = mlines.Line2D([], [], color="#555555", marker="o", linestyle="None",
                          markersize=6, label="Individual observations")
dot_mean  = mlines.Line2D([], [], color="#111111", marker="D", linestyle="None",
                          markersize=6, label="Mean")
fig1.legend(
    handles=[patch_left, patch_right, patch_mid, dot_obs, dot_mean],
    loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0),
    frameon=True, framealpha=0.9, edgecolor="#cccccc", fontsize=10
)

fig1.tight_layout(rect=[0, 0.04, 1, 0.97])
fig1.savefig("fig1_boxplots_2x4.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig1)
print("\nSaved: fig1_boxplots_2x4.png")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — PER-TARGET MEAN ERROR BAR CHART
# Both sheets plotted side-by-side for each metric, only common targets.
# Gives an immediate visual of which targets perform better/worse under each
# experimental condition.
# ──────────────────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(2, 2, figsize=(26, 16), sharey=True)
fig2.patch.set_facecolor(WHITE)
fig2.suptitle(
    "Per-Target Mean Error: adjusted_position vs. adapter\n(common targets, averaged across operators)",
    fontsize=17, fontweight="bold"
)

x = np.arange(len(common_targets))
w = 0.38  # bar width
common_group_spans = get_target_group_spans(common_targets, combined_target_side)
tick_rotation_common, tick_align_common = target_tick_rotation(len(common_targets))
fig2_shared_ylim = compute_shared_ylim(
    [adj_common[m].to_numpy(dtype=float) for m in METRICS] +
    [ada_common[m].to_numpy(dtype=float) for m in METRICS],
    include_zero=True,
)

for ax, metric in zip(axes2.flatten(), METRICS):
    style_axis(ax)
    for start_idx, end_idx, side_label in common_group_spans:
        ax.axvspan(start_idx - 0.5, end_idx + 0.5,
                   color=SIDE_COLORS.get(side_label, "#6C757D"), alpha=0.05, zorder=0)

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

    apply_shared_y_axis(ax, fig2_shared_ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(common_targets, rotation=tick_rotation_common, ha=tick_align_common)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric], fontweight="bold", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    for start_idx, end_idx, side_label in common_group_spans:
        midpoint = (start_idx + end_idx) / 2
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(midpoint, 1.02, side_label, transform=trans,
                ha="center", va="bottom", fontsize=9,
                color=SIDE_COLORS.get(side_label, "#6C757D"), fontweight="bold")

fig2.tight_layout()
fig2.savefig("fig2_per_target_bar.png", bbox_inches="tight", facecolor=WHITE)
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
    subplot_kw={"polar": True},
    squeeze=False,
)
fig3.patch.set_facecolor(WHITE)
fig3.suptitle(
    "Radar Error Profiles per Target (common targets)\n"
    "Top: adjusted_position  |  Bottom: adapter",
    fontsize=17, fontweight="bold"
)

fig3_radar_limit = max(
    float(
        np.nanmax(
            np.vstack([
                avg_adj[radar_metrics].to_numpy(dtype=float),
                avg_ada[radar_metrics].to_numpy(dtype=float),
            ])
        )
    ) * 1.2,
    0.5,
)

for j, (df_src, sheet) in enumerate([
    (avg_adj, "adjusted_position"),
    (avg_ada, "adapter"),
]):
    for i, t in enumerate(common_targets):
        ax = axes3[j][i]
        ax.set_facecolor(WHITE)
        row = df_src[df_src["target_id"] == t]
        if row.empty:
            ax.set_visible(False)
            continue
        vals = row[radar_metrics].values.flatten().tolist()
        color = SHEET_COLORS[sheet]
        draw_radar(ax, vals, color, sheet)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_radar, fontsize=10)
        side_label = combined_target_side.get(t, "Other / Midline")
        ax.set_title(f"{t} ({side_label})\n{sheet}", fontsize=9, pad=12,
                 color=SIDE_COLORS.get(side_label, color))
        ax.set_ylim(0, fig3_radar_limit)
        ax.yaxis.set_tick_params(labelsize=8, colors=TEXT_COLOR)
        ax.grid(True, alpha=0.35, color=GRID_COLOR)

fig3.tight_layout()
fig3.savefig("fig3_radar_per_target.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig3)
print("Saved: fig3_radar_per_target.png")

if 2 > n_targets:
    fig3t, axes3t = plt.subplots(
        n_targets,
        2,
        figsize=(18, 9 * n_targets),
        subplot_kw={"polar": True},
        squeeze=False,
    )
    fig3t.patch.set_facecolor(WHITE)
    fig3t.suptitle(
        "Radar Error Profiles per Target (common targets, transposed layout)",
        fontsize=17, fontweight="bold"
    )

    for col_idx, (df_src, sheet) in enumerate([
        (avg_adj, "adjusted_position"),
        (avg_ada, "adapter"),
    ]):
        for row_idx, t in enumerate(common_targets):
            ax = axes3t[row_idx][col_idx]
            ax.set_facecolor(WHITE)
            row = df_src[df_src["target_id"] == t]
            if row.empty:
                ax.set_visible(False)
                continue
            vals = row[radar_metrics].values.flatten().tolist()
            color = SHEET_COLORS[sheet]
            draw_radar(ax, vals, color, sheet)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels_radar, fontsize=10)
            side_label = combined_target_side.get(t, "Other / Midline")
            ax.set_title(f"{t} ({side_label})\n{sheet}", fontsize=10, pad=12,
                         color=SIDE_COLORS.get(side_label, color))
            ax.set_ylim(0, fig3_radar_limit)
            ax.yaxis.set_tick_params(labelsize=8, colors=TEXT_COLOR)
            ax.grid(True, alpha=0.35, color=GRID_COLOR)

    fig3t.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig3t.savefig("fig3_radar_per_target_transposed.png", bbox_inches="tight", facecolor=WHITE)
    plt.close(fig3t)
    print("Saved: fig3_radar_per_target_transposed.png")

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
fig4.patch.set_facecolor(WHITE)
fig4.suptitle(
    "Heat-Maps of Error by Target × Operator\nTop: adjusted_position  |  Bottom: adapter",
    fontsize=17, fontweight="bold"
)

def make_heatmap_matrix(df, metric):
    """Return pivot matrix (targets as rows, operators as columns)."""
    return df.pivot_table(index="target_id", columns="operator",
                          values=metric, aggfunc="mean")


heatmap_cache = {}
heatmap_values = []
for metric in METRICS:
    for df_src, sheet in [
        (df_adj, "adjusted_position"),
        (df_ada, "adapter"),
    ]:
        side_map = target_side_adj if sheet == "adjusted_position" else target_side_ada
        ordered_targets = get_ordered_targets(make_heatmap_matrix(df_src, metric).index.tolist(), side_map)
        mat = make_heatmap_matrix(df_src, metric).reindex(index=ordered_targets)
        heatmap_cache[(sheet, metric)] = mat
        heatmap_values.append(mat.to_numpy(dtype=float).ravel())

all_heatmap_vals = np.concatenate(heatmap_values)
all_heatmap_vals = all_heatmap_vals[~np.isnan(all_heatmap_vals)]
heatmap_vmin = float(np.min(all_heatmap_vals)) if all_heatmap_vals.size else 0.0
heatmap_vmax = float(np.max(all_heatmap_vals)) if all_heatmap_vals.size else 1.0

for col_idx, metric in enumerate(METRICS):
    for row_idx, (df_src, sheet) in enumerate([
        (df_adj, "adjusted_position"),
        (df_ada, "adapter"),
    ]):
        ax = axes4[row_idx][col_idx]
        style_axis(ax)
        mat = heatmap_cache[(sheet, metric)]
        im = ax.imshow(mat.values, cmap=cmap_custom, aspect="auto", vmin=heatmap_vmin, vmax=heatmap_vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"Op {c}" for c in mat.columns], fontsize=10)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=10)
        ax.set_title(f"{METRIC_LABELS[metric]}\n({sheet})", fontsize=12, fontweight="bold")

        # Annotate each cell with its value
        for (i_r, i_c), val in np.ndenumerate(mat.values):
            if not np.isnan(val):
                ax.text(i_c, i_r, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color=heatmap_label_color(float(val), heatmap_vmin, heatmap_vmax))
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

fig4.tight_layout()
fig4.savefig("fig4_heatmaps.png", bbox_inches="tight", facecolor=WHITE)
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
fig5.patch.set_facecolor(WHITE)
fig5.suptitle("Pearson Correlation Matrix of Error Metrics",
              fontsize=17, fontweight="bold")

def draw_corr_heatmap(ax, corr_df, title):
    style_axis(ax)
    mat = corr_df.values
    im  = ax.imshow(mat, cmap=cmap_corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(short_labels)));  ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_yticks(range(len(short_labels)));  ax.set_yticklabels(short_labels, fontsize=11)
    ax.set_title(title, fontweight="bold")
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="black" if abs(val) < 0.7 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)

draw_corr_heatmap(ax5a, corr_adj, "adjusted_position")
draw_corr_heatmap(ax5b, corr_ada, "adapter")

fig5.tight_layout()
fig5.savefig("fig5_correlations.png", bbox_inches="tight", facecolor=WHITE)
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
fig8.patch.set_facecolor(WHITE)
fig8.suptitle(
    "Stacked Error Contributions per Target (averaged across operators)\n"
    "Left: adjusted_position  |  Right: adapter",
    fontsize=17, fontweight="bold"
)

component_metrics = ["error_x", "error_y", "error_z"]
component_colors  = [METRIC_COLORS[m] for m in component_metrics]
x8 = np.arange(len(common_targets))

def stacked_bar(ax, df_avg, title):
    style_axis(ax)
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

    tick_rotation, tick_align = target_tick_rotation(len(common_targets))
    ax.set_xticks(x8)
    ax.set_xticklabels(common_targets, rotation=tick_rotation, ha=tick_align)
    ax.set_ylabel("Error (mm)")
    ax.set_title(title, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right")

stacked_bar(ax8a, avg_adj[avg_adj["target_id"].isin(common_targets)], "adjusted_position")
stacked_bar(ax8b, avg_ada[avg_ada["target_id"].isin(common_targets)], "adapter (filtered)")
fig8_shared_ylim = compute_shared_ylim(
    [avg_adj[component_metrics].to_numpy(dtype=float).ravel(), avg_ada[component_metrics].to_numpy(dtype=float).ravel()],
    include_zero=True,
)
apply_shared_y_axis(ax8a, fig8_shared_ylim)
apply_shared_y_axis(ax8b, fig8_shared_ylim)

fig8.tight_layout()
fig8.savefig("fig8_stacked_contributions.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig8)
print("Saved: fig8_stacked_contributions.png")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — OVERALL ERROR DISTRIBUTION: 2 × 1 BOX PLOTS
# Top: adjusted_position  |  Bottom: adapter
# Each subplot shows one box per error metric pooled across all targets,
# giving a compact single-panel summary of the four error axes per setup.
# ──────────────────────────────────────────────────────────────────────────────

fig9, axes9 = plt.subplots(2, 1, figsize=(20, 24), sharey=True,
                            gridspec_kw={"hspace": 0.38})
fig9.patch.set_facecolor(WHITE)
fig9.suptitle(
    "Overall Error Distribution by Metric\n(all targets pooled)",
    fontsize=17, fontweight="bold"
)

rng9 = np.random.default_rng(42)
fig9_shared_ylim = compute_shared_ylim(
    [df_adj[df_adj["target_id"].isin(common_targets)][m].dropna().to_numpy(dtype=float) for m in METRICS] +
    [df_ada[df_ada["target_id"].isin(common_targets)][m].dropna().to_numpy(dtype=float) for m in METRICS]
)

for ax, (df_src, sheet_name) in zip(axes9, [
    (df_adj, "adjusted_position"),
    (df_ada, "adapter"),
]):
    style_axis(ax)
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

    apply_shared_y_axis(ax, fig9_shared_ylim)
    ax.tick_params(axis="y", which="major", labelsize=10, length=5, width=0.9)
    ax.tick_params(axis="y", which="minor", length=3, width=0.6)

    y_lo2, _ = ax.get_ylim()
    for pos, data in zip(range(1, len(METRICS) + 1), groups):
        ax.text(pos, y_lo2, f"n={len(data)}", ha="center", va="bottom",
                fontsize=9, color="gray")

fig9.tight_layout(rect=[0, 0.01, 1, 0.96])
fig9.savefig("fig9_overall_boxplots_2x1.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig9)
print("Saved: fig9_overall_boxplots_2x1.png")

save_transposed_overall_boxplots("fig9_overall_boxplots_1x2.png", fig9_shared_ylim)

# ──────────────────────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────────────────────
print("\n✔  All figures saved. Analysis complete.")
