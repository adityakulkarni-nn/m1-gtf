"""
=============================================================================
adjusted_position_all_targets_analysis.py
=============================================================================
Focused statistical analysis and visualisation of targeting error data from the
`adjusted_position` sheet in `master_error.xlsx`.

This script keeps ALL available targets from the sheet and produces a compact
set of figures tailored to the adjusted-position experiment only.

Outputs
-------
1. adjusted_position_fig1_boxplots_all_targets.png
   Distribution of each error metric by target.
2. adjusted_position_fig2_mean_sd_bars.png
   Per-target mean ± SD across the four metrics.
3. adjusted_position_fig3_operator_heatmaps.png
   Target × operator heatmaps for each metric.
4. adjusted_position_fig4_target_profiles.png
   Mean error profile of every target across all metrics.
5. adjusted_position_fig5_correlations.png
   Correlation matrix of the four error metrics.
6. adjusted_position_fig6_operator_total_error.png
   Operator-level total error comparison with target jitter.

Additionally, summary tables are saved as:
- adjusted_position_target_summary.csv
- adjusted_position_operator_summary.csv
=============================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


# ──────────────────────────────────────────────────────────────────────────────
# 0. PATHS / STYLE
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FILE = SCRIPT_DIR / "master_error.xlsx"
OUTPUT_DIR = SCRIPT_DIR

SHEET_NAME = "adjusted_position"
METRICS = ["error_x", "error_y", "error_z", "total_error"]
METRIC_LABELS = {
    "error_x": "Error X (mm)",
    "error_y": "Error Y (mm)",
    "error_z": "Error Z (mm)",
    "total_error": "Total Error (mm)",
}
METRIC_COLORS = {
    "error_x": "#4C72B0",
    "error_y": "#DD8452",
    "error_z": "#55A868",
    "total_error": "#C44E52",
}
PRIMARY_COLOR = "#2E86AB"
SECONDARY_COLOR = "#174C72"
ACCENT_COLOR = "#F18F01"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING / CLEANING
# ──────────────────────────────────────────────────────────────────────────────
if not FILE.exists():
    raise FileNotFoundError(f"Could not find workbook: {FILE}")

df = pd.read_excel(FILE, sheet_name=SHEET_NAME).copy()
df["operator"] = df["operator"].astype(str)

missing_metrics = [metric for metric in METRICS if metric not in df.columns]
if missing_metrics:
    raise KeyError(f"Missing expected metric columns: {missing_metrics}")

# keep only rows with at least one valid metric
metric_mask = df[METRICS].notna().any(axis=1)
df = df.loc[metric_mask].copy()

all_targets = sorted(df["target_id"].dropna().unique().tolist())
all_operators = sorted(df["operator"].dropna().unique().tolist())

print(f"Loaded sheet               : {SHEET_NAME}")
print(f"Workbook                   : {FILE.name}")
print(f"Targets included (all)     : {all_targets}")
print(f"Operators included         : {all_operators}")
print(f"Number of observations     : {len(df)}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. SUMMARIES
# ──────────────────────────────────────────────────────────────────────────────
def coefficient_of_variation(series: pd.Series) -> float:
    mean = series.mean()
    std = series.std(ddof=1)
    if pd.isna(mean) or np.isclose(mean, 0):
        return np.nan
    return (std / mean) * 100


def nice_step(span: float) -> float:
    span = max(float(span), 0.1)
    raw_step = span / 6
    magnitude = 10 ** np.floor(np.log10(raw_step))
    nice_steps = np.array([0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0])
    normalized = raw_step / magnitude
    return float(magnitude * nice_steps[np.argmin(np.abs(nice_steps - normalized))])


def annotate_median(ax: plt.Axes, groups: list[np.ndarray], positions: list[int]) -> None:
    for pos, group in zip(positions, groups):
        clean = np.asarray(group, dtype=float)
        clean = clean[~np.isnan(clean)]
        if clean.size == 0:
            continue
        median = np.median(clean)
        ax.text(
            pos,
            median,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
            fontweight="bold",
        )


def apply_fine_y_axis(ax: plt.Axes, data_arrays: list[np.ndarray]) -> None:
    clean_arrays = [np.asarray(arr, dtype=float) for arr in data_arrays if len(arr) > 0]
    if not clean_arrays:
        return
    all_vals = np.concatenate(clean_arrays)
    all_vals = all_vals[~np.isnan(all_vals)]
    if all_vals.size == 0:
        return

    data_min, data_max = float(all_vals.min()), float(all_vals.max())
    span = max(data_max - data_min, 0.1)
    pad = span * 0.12
    ax.set_ylim(data_min - pad, data_max + pad * 2.2)
    y_lo, y_hi = ax.get_ylim()
    ax.yaxis.set_major_locator(MultipleLocator(nice_step(y_hi - y_lo)))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)


def corr_label_color(value: float) -> str:
    return "white" if abs(value) >= 0.7 else "black"


descriptive = df[METRICS].describe().T
for metric in METRICS:
    descriptive.loc[metric, "cv_%"] = round(coefficient_of_variation(df[metric]), 2)

print("\n" + "=" * 68)
print("DESCRIPTIVE STATISTICS — ADJUSTED POSITION")
print("=" * 68)
print(descriptive.round(3).to_string())

target_summary = (
    df.groupby("target_id")[METRICS]
    .agg(["mean", "std", "median", "min", "max", "count"])
    .round(3)
)
target_summary.columns = [f"{metric}_{stat}" for metric, stat in target_summary.columns]
target_summary = target_summary.reset_index()

target_mean = (
    df.groupby("target_id")[METRICS]
    .mean()
    .reindex(all_targets)
    .reset_index()
)

operator_summary = (
    df.groupby("operator")[METRICS]
    .agg(["mean", "std", "median", "count"])
    .round(3)
)
operator_summary.columns = [f"{metric}_{stat}" for metric, stat in operator_summary.columns]
operator_summary = operator_summary.reset_index()

print("\n--- Per-target summary ---")
print(target_summary.to_string(index=False))
print("\n--- Per-operator summary ---")
print(operator_summary.to_string(index=False))

target_summary.to_csv(OUTPUT_DIR / "adjusted_position_target_summary.csv", index=False)
operator_summary.to_csv(OUTPUT_DIR / "adjusted_position_operator_summary.csv", index=False)
print("\nSaved: adjusted_position_target_summary.csv")
print("Saved: adjusted_position_operator_summary.csv")


# ──────────────────────────────────────────────────────────────────────────────
# 3. FIGURE 1 — BOXPLOTS BY TARGET FOR ALL METRICS
# ──────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(24, 14))
fig1.patch.set_facecolor("#fafafa")
fig1.suptitle(
    "Adjusted Position: Error Distribution by Target (all targets)",
    fontsize=15,
    fontweight="bold",
)

rng = np.random.default_rng(42)

for ax, metric in zip(axes1.flatten(), METRICS):
    groups = [df.loc[df["target_id"] == target, metric].dropna().values for target in all_targets]
    positions = list(range(1, len(all_targets) + 1))

    box = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.5,
        whis=(0, 100),
        showfliers=False,
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(color=PRIMARY_COLOR, linewidth=1.3, linestyle="--"),
        capprops=dict(color=PRIMARY_COLOR, linewidth=1.6),
        boxprops=dict(linewidth=1.2, color=PRIMARY_COLOR),
    )
    for patch in box["boxes"]:
        patch.set_facecolor(METRIC_COLORS[metric])
        patch.set_alpha(0.65)

    for pos, values in zip(positions, groups):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(
            pos + jitter,
            values,
            s=34,
            color=SECONDARY_COLOR,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    annotate_median(ax, groups, positions)
    apply_fine_y_axis(ax, groups)

    ax.set_xticks(positions)
    ax.set_xticklabels(all_targets, rotation=35, ha="right")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric], fontweight="bold")

    y_lo, _ = ax.get_ylim()
    for pos, values in zip(positions, groups):
        ax.text(pos, y_lo, f"n={len(values)}", ha="center", va="bottom", fontsize=8, color="gray")

fig1.tight_layout(rect=[0, 0.02, 1, 0.96])
fig1.savefig(OUTPUT_DIR / "adjusted_position_fig1_boxplots_all_targets.png", bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close(fig1)
print("Saved: adjusted_position_fig1_boxplots_all_targets.png")


# ──────────────────────────────────────────────────────────────────────────────
# 4. FIGURE 2 — PER-TARGET MEAN ± SD BARS
# ──────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(24, 14))
fig2.suptitle(
    "Adjusted Position: Per-Target Mean ± SD",
    fontsize=15,
    fontweight="bold",
)

x = np.arange(len(all_targets))

for ax, metric in zip(axes2.flatten(), METRICS):
    grouped = df.groupby("target_id")[metric]
    means = grouped.mean().reindex(all_targets)
    stds = grouped.std().reindex(all_targets).fillna(0)

    bars = ax.bar(
        x,
        means.values,
        yerr=stds.values,
        capsize=5,
        color=METRIC_COLORS[metric],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )

    for bar, value in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds.max(), 0.02) * 0.2 + 0.01,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_targets)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric], fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

fig2.tight_layout(rect=[0, 0.02, 1, 0.96])
fig2.savefig(OUTPUT_DIR / "adjusted_position_fig2_mean_sd_bars.png", bbox_inches="tight")
plt.close(fig2)
print("Saved: adjusted_position_fig2_mean_sd_bars.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 3 — TARGET × OPERATOR HEATMAPS
# ──────────────────────────────────────────────────────────────────────────────
cmap_heat = LinearSegmentedColormap.from_list(
    "adjusted_err_cmap", ["#d8f3dc", "#ffd166", "#ef476f"]
)

fig3, axes3 = plt.subplots(2, 2, figsize=(22, 14))
fig3.suptitle(
    "Adjusted Position: Target × Operator Error Heatmaps",
    fontsize=15,
    fontweight="bold",
)

for ax, metric in zip(axes3.flatten(), METRICS):
    matrix = (
        df.pivot_table(index="target_id", columns="operator", values=metric, aggfunc="mean")
        .reindex(index=all_targets, columns=all_operators)
    )
    im = ax.imshow(matrix.values, cmap=cmap_heat, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"Op {op}" for op in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(METRIC_LABELS[metric], fontweight="bold")

    for (row_idx, col_idx), value in np.ndenumerate(matrix.values):
        if not np.isnan(value):
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

fig3.tight_layout(rect=[0, 0.02, 1, 0.96])
fig3.savefig(OUTPUT_DIR / "adjusted_position_fig3_operator_heatmaps.png", bbox_inches="tight")
plt.close(fig3)
print("Saved: adjusted_position_fig3_operator_heatmaps.png")


# ──────────────────────────────────────────────────────────────────────────────
# 6. FIGURE 4 — TARGET PROFILES ACROSS METRICS
# ──────────────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(16, 9))
fig4.suptitle(
    "Adjusted Position: Mean Error Profile per Target",
    fontsize=15,
    fontweight="bold",
)

profile_x = np.arange(len(METRICS))
for idx, target in enumerate(all_targets):
    row = target_mean[target_mean["target_id"] == target]
    if row.empty:
        continue
    values = row[METRICS].iloc[0].values.astype(float)
    ax4.plot(
        profile_x,
        values,
        marker="o",
        linewidth=2,
        markersize=6,
        alpha=0.85,
        label=str(target),
    )
    ax4.text(profile_x[-1] + 0.05, values[-1], str(target), fontsize=8, va="center")

ax4.set_xticks(profile_x)
ax4.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
ax4.set_ylabel("Mean error (mm)")
ax4.grid(True, linestyle="--", alpha=0.35)
ax4.set_axisbelow(True)

fig4.tight_layout()
fig4.savefig(OUTPUT_DIR / "adjusted_position_fig4_target_profiles.png", bbox_inches="tight")
plt.close(fig4)
print("Saved: adjusted_position_fig4_target_profiles.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. FIGURE 5 — CORRELATION MATRIX
# ──────────────────────────────────────────────────────────────────────────────
corr = df[METRICS].corr()
short_labels = ["X", "Y", "Z", "Total"]
cmap_corr = LinearSegmentedColormap.from_list(
    "corr_cmap", ["#2E86AB", "#FFFFFF", "#E84855"]
)

fig5, ax5 = plt.subplots(figsize=(9, 8))
fig5.suptitle(
    "Adjusted Position: Pearson Correlation Matrix",
    fontsize=15,
    fontweight="bold",
)

im = ax5.imshow(corr.values, cmap=cmap_corr, vmin=-1, vmax=1, aspect="equal")
ax5.set_xticks(range(len(short_labels)))
ax5.set_xticklabels(short_labels)
ax5.set_yticks(range(len(short_labels)))
ax5.set_yticklabels(short_labels)

for (i, j), value in np.ndenumerate(corr.values):
    ax5.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11, color=corr_label_color(value))

plt.colorbar(im, ax=ax5, shrink=0.85)
fig5.tight_layout()
fig5.savefig(OUTPUT_DIR / "adjusted_position_fig5_correlations.png", bbox_inches="tight")
plt.close(fig5)
print("Saved: adjusted_position_fig5_correlations.png")


# ──────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 6 — OPERATOR COMPARISON FOR TOTAL ERROR
# ──────────────────────────────────────────────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(12, 8))
fig6.suptitle(
    "Adjusted Position: Operator Comparison for Total Error",
    fontsize=15,
    fontweight="bold",
)

operator_groups = [
    df.loc[df["operator"] == operator, "total_error"].dropna().values
    for operator in all_operators
]
positions = list(range(1, len(all_operators) + 1))

box6 = ax6.boxplot(
    operator_groups,
    patch_artist=True,
    widths=0.5,
    whis=(0, 100),
    showfliers=False,
    medianprops=dict(color="white", linewidth=2.2),
    whiskerprops=dict(color=SECONDARY_COLOR, linewidth=1.3, linestyle="--"),
    capprops=dict(color=SECONDARY_COLOR, linewidth=1.6),
    boxprops=dict(linewidth=1.2, color=SECONDARY_COLOR),
)
for patch in box6["boxes"]:
    patch.set_facecolor(PRIMARY_COLOR)
    patch.set_alpha(0.75)

for pos, values in zip(positions, operator_groups):
    if len(values) == 0:
        continue
    jitter = rng.uniform(-0.14, 0.14, size=len(values))
    ax6.scatter(
        pos + jitter,
        values,
        color=ACCENT_COLOR,
        s=40,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

annotate_median(ax6, operator_groups, positions)
apply_fine_y_axis(ax6, operator_groups)
ax6.set_xticks(positions)
ax6.set_xticklabels([f"Operator {op}" for op in all_operators])
ax6.set_ylabel("Total Error (mm)")
ax6.set_title("Pooled across all targets", fontweight="bold")

fig6.tight_layout()
fig6.savefig(OUTPUT_DIR / "adjusted_position_fig6_operator_total_error.png", bbox_inches="tight")
plt.close(fig6)
print("Saved: adjusted_position_fig6_operator_total_error.png")

print("\n✔ Adjusted-position analysis complete.")
