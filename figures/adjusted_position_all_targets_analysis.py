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
1b. adjusted_position_fig1b_total_error_boxplot.png
    Manuscript-ready total error distribution by target.
2. adjusted_position_fig2_mean_sd_bars.png
   Per-target mean ± SD across the four metrics.
3. adjusted_position_fig3_operator_heatmaps.png
   Target × operator heatmaps for each metric.
4. adjusted_position_fig4_target_profiles.png
    Mean error profile of every target across all metrics, grouped by side.
5. adjusted_position_fig5_target_radars.png
    Radar plots showing the mean error profile for each target.
6. adjusted_position_fig6_average_error_radar.png
    Radar plot showing the overall average error for X, Y, Z, and total error.
7. adjusted_position_fig7_correlations.png
   Correlation matrix of the four error metrics.
8. adjusted_position_fig8_operator_total_error.png
   Operator-level total error comparison with target jitter.
9. adjusted_position_fig9_operator_rmse.png
    Per-target interoperator RMSE with global pooled reference lines.
10. adjusted_position_fig10_orientation_effects.png
     Effect of collar and arc angle changes on Euclidean targeting error.

Additionally, summary tables are saved as:
- adjusted_position_target_summary.csv
- adjusted_position_operator_summary.csv
- adjusted_position_interoperator_rmse_by_target.csv
- adjusted_position_interoperator_rmse_global.csv
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
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
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
WHITE = "#FFFFFF"
GRID_COLOR = "#D7DEE8"
TEXT_COLOR = "#1F2933"

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "legend.fontsize": 12,
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


def rmse(series: pd.Series) -> float:
    clean = series.dropna().astype(float)
    if clean.empty:
        return np.nan
    return float(np.sqrt(np.mean(np.square(clean))))


def one_way_anova_f(groups: list[np.ndarray]) -> float:
    clean_groups = []
    for group in groups:
        arr = np.asarray(group, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size > 0:
            clean_groups.append(arr)

    if len(clean_groups) < 2:
        return np.nan

    all_values = np.concatenate(clean_groups)
    grand_mean = float(all_values.mean())
    ss_between = sum(len(group) * (float(group.mean()) - grand_mean) ** 2 for group in clean_groups)
    ss_within = sum(float(((group - group.mean()) ** 2).sum()) for group in clean_groups)
    df_between = len(clean_groups) - 1
    df_within = len(all_values) - len(clean_groups)
    if df_between <= 0 or df_within <= 0 or np.isclose(ss_within, 0):
        return np.nan
    return float((ss_between / df_between) / (ss_within / df_within))


def permutation_anova_pvalue(groups: list[np.ndarray], n_permutations: int = 20000, seed: int = 42) -> tuple[float, float]:
    clean_groups = []
    for group in groups:
        arr = np.asarray(group, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size > 0:
            clean_groups.append(arr)

    observed_f = one_way_anova_f(clean_groups)
    if len(clean_groups) < 2 or np.isnan(observed_f):
        return np.nan, np.nan

    group_sizes = [len(group) for group in clean_groups]
    combined = np.concatenate(clean_groups)
    rng_local = np.random.default_rng(seed)
    exceed_count = 0

    for _ in range(n_permutations):
        shuffled = rng_local.permutation(combined)
        permuted_groups = []
        start = 0
        for size in group_sizes:
            permuted_groups.append(shuffled[start:start + size])
            start += size
        permuted_f = one_way_anova_f(permuted_groups)
        if not np.isnan(permuted_f) and permuted_f >= observed_f:
            exceed_count += 1

    p_value = (exceed_count + 1) / (n_permutations + 1)
    return observed_f, float(p_value)


def nice_step(span: float) -> float:
    span = max(float(span), 0.1)
    raw_step = span / 6
    magnitude = 10 ** np.floor(np.log10(raw_step))
    nice_steps = np.array([0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0])
    normalized = raw_step / magnitude
    return float(magnitude * nice_steps[np.argmin(np.abs(nice_steps - normalized))])


def target_tick_rotation(count: int) -> tuple[int, str]:
    if count >= 10:
        return 35, "right"
    return 0, "center"


def compute_shared_ylim(
    data_arrays: list[np.ndarray],
    *,
    include_zero: bool = False,
    top_padding: float = 0.18,
    bottom_padding: float = 0.10,
) -> tuple[float, float]:
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


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(WHITE)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", labelsize=12, colors=TEXT_COLOR)


def apply_shared_y_axis(ax: plt.Axes, y_limits: tuple[float, float]) -> None:
    y_lo, y_hi = y_limits
    if np.isclose(y_lo, y_hi):
        y_hi = y_lo + 1.0
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(MultipleLocator(nice_step(y_hi - y_lo)))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.7, color=GRID_COLOR)
    ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.35, color=GRID_COLOR)
    ax.set_axisbelow(True)


def heatmap_label_color(value: float, vmin: float, vmax: float) -> str:
    if np.isnan(value) or np.isclose(vmax, vmin):
        return "black"
    midpoint = (vmax + vmin) / 2
    return "white" if value >= midpoint else TEXT_COLOR


def render_target_radars(
    n_rows: int,
    n_cols: int,
    output_name: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 6 * n_rows),
        subplot_kw={"polar": True},
        squeeze=False,
    )
    fig.patch.set_facecolor(WHITE)
    fig.suptitle(title, fontsize=17, fontweight="bold")

    for ax5, (_, row) in zip(axes.flatten(), target_mean.iterrows()):
        target_label = str(row["target_id"])
        values = row[METRICS].values.astype(float).tolist()
        color = side_color(str(row["side"]))
        ax5.set_facecolor(WHITE)
        plot_radar(ax5, values, color=color, label=target_label)
        annotate_radar_values(ax5, values)
        ax5.set_xticks(RADAR_ANGLES[:-1])
        ax5.set_xticklabels(RADAR_LABELS, fontsize=10)
        ax5.set_ylim(0, radar_limit)
        ax5.set_title(f"{target_label} ({row['side']})", fontsize=12, fontweight="bold", pad=18)
        ax5.grid(True, alpha=0.35, color=GRID_COLOR)
        ax5.tick_params(axis="y", labelsize=9, colors=TEXT_COLOR)

    for ax5 in axes.flatten()[n_targets:]:
        ax5.set_visible(False)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_DIR / output_name, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved: {output_name}")


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


def normalize_side_label(side: object) -> str:
    text = str(side).strip().upper()
    if text == "L" or text == "LEFT":
        return "Left"
    if text == "R" or text == "RIGHT":
        return "Right"
    if text == "M" or text == "MIDLINE" or text == "CENTER":
        return "Other / Midline"
    return "Other / Midline"


def resolve_target_side(side_values: pd.Series) -> str:
    normalized = side_values.dropna().map(normalize_side_label)
    if normalized.empty:
        return "Other / Midline"

    preferred = normalized[normalized.isin(["Left", "Right"])]
    if not preferred.empty:
        return str(preferred.mode().iloc[0])

    return str(normalized.mode().iloc[0])


def side_color(side: str) -> str:
    return {
        "Left": PRIMARY_COLOR,
        "Right": ACCENT_COLOR,
        "Other / Midline": "#6C757D",
    }.get(side, SECONDARY_COLOR)


RADAR_LABELS = ["Error X", "Error Y", "Error Z", "Total\nError"]
RADAR_ANGLES = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
RADAR_ANGLES += RADAR_ANGLES[:1]


def plot_radar(ax: plt.Axes, values: list[float], color: str, label: str | None = None, fill_alpha: float = 0.18) -> None:
    closed_values = values + values[:1]
    ax.plot(RADAR_ANGLES, closed_values, color=color, linewidth=2.2, marker="o", label=label)
    ax.fill(RADAR_ANGLES, closed_values, color=color, alpha=fill_alpha)


def annotate_radar_values(ax: plt.Axes, values: list[float]) -> None:
    for angle, value in zip(RADAR_ANGLES[:-1], values):
        ax.text(angle, value + 0.04, f"{value:.2f}", ha="center", va="center", fontsize=8)


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
target_side = (
    df.groupby("target_id")["side"]
    .apply(resolve_target_side)
    .reindex(all_targets)
)
target_mean["side"] = target_mean["target_id"].map(target_side)
target_side_map = target_side.to_dict()
side_order = {"Left": 0, "Right": 1, "Other / Midline": 2}
ordered_targets = sorted(
    all_targets,
    key=lambda target: (side_order.get(target_side_map.get(target, "Other / Midline"), 99), str(target)),
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

target_summary.to_csv(OUTPUT_DIR / f"{SHEET_NAME}_target_summary.csv", index=False)
operator_summary.to_csv(OUTPUT_DIR / f"{SHEET_NAME}_operator_summary.csv", index=False)
print(f"\nSaved: {SHEET_NAME}_target_summary.csv")
print(f"Saved: {SHEET_NAME}_operator_summary.csv\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. FIGURE 1 — BOXPLOTS BY TARGET FOR ALL METRICS
# ──────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(26, 15), sharey=True)
fig1.patch.set_facecolor(WHITE)
fig1.suptitle(
    "Adjusted Position: Error Distribution by Target",
    fontsize=17,
    fontweight="bold",
)

shared_fig1_ylim = compute_shared_ylim([df[metric].dropna().to_numpy(dtype=float) for metric in METRICS])

rng = np.random.default_rng(42)

target_group_spans: list[tuple[int, int, str]] = []
span_start = 0
current_side = target_side_map.get(ordered_targets[0], "Other / Midline")
for idx, target in enumerate(ordered_targets[1:], start=1):
    target_side_label = target_side_map.get(target, "Other / Midline")
    if target_side_label != current_side:
        target_group_spans.append((span_start, idx - 1, current_side))
        span_start = idx
        current_side = target_side_label
target_group_spans.append((span_start, len(ordered_targets) - 1, current_side))


def draw_target_distribution_panel(ax: plt.Axes, metric: str, y_limits: tuple[float, float] | None = None) -> None:
    groups = [df.loc[df["target_id"] == target, metric].dropna().values for target in ordered_targets]
    positions = list(range(1, len(ordered_targets) + 1))
    tick_rotation, tick_align = target_tick_rotation(len(ordered_targets))

    style_axis(ax)

    for start_idx, end_idx, side_label in target_group_spans:
        ax.axvspan(start_idx + 0.5, end_idx + 1.5, color=side_color(side_label), alpha=0.06, zorder=0)

    box = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.5,
        whis=(0, 100),
        showfliers=False,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="#111111", markeredgecolor="white", markersize=6),
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(color=PRIMARY_COLOR, linewidth=1.3, linestyle="--"),
        capprops=dict(color=PRIMARY_COLOR, linewidth=1.6),
        boxprops=dict(linewidth=1.2, color=PRIMARY_COLOR),
    )
    for patch, target in zip(box["boxes"], ordered_targets):
        patch.set_facecolor(side_color(target_side_map.get(target, "Other / Midline")))
        patch.set_alpha(0.65)

    for pos, values, target in zip(positions, groups, ordered_targets):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        point_color = side_color(target_side_map.get(target, "Other / Midline"))
        ax.scatter(
            pos + jitter,
            values,
            s=34,
            color=point_color,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    if y_limits is None:
        apply_fine_y_axis(ax, groups)
    else:
        apply_shared_y_axis(ax, y_limits)

    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_targets, rotation=tick_rotation, ha=tick_align)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric], fontweight="bold", fontsize=13)
    ax.grid(True, axis="x", linestyle=":", alpha=0.20, color=GRID_COLOR)
    ax.tick_params(axis="x", labelsize=10, pad=8)

    y_lo, y_hi = ax.get_ylim()
    y_text = y_hi + (y_hi - y_lo) * 0.03
    for start_idx, end_idx, side_label in target_group_spans:
        midpoint = (positions[start_idx] + positions[end_idx]) / 2
        ax.text(
            midpoint,
            y_text,
            side_label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=side_color(side_label),
            clip_on=False,
        )


legend_handles = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor=side_color("Left"), markeredgecolor="none", markersize=11, label="Left targets"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=side_color("Right"), markeredgecolor="none", markersize=11, label="Right targets"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=side_color("Other / Midline"), markeredgecolor="none", markersize=11, label="Other / midline"),
    Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555", linestyle="None", markersize=6, label="Individual observations"),
    Line2D([0], [0], marker="D", color="#111111", markerfacecolor="#111111", linestyle="None", markersize=6, label="Mean"),
]

for ax, metric in zip(axes1.flatten(), METRICS):
    draw_target_distribution_panel(ax, metric, shared_fig1_ylim)

fig1.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.01))

fig1.tight_layout(rect=[0, 0.06, 1, 0.95])
fig1.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig1_boxplots_all_targets.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig1)
print(f"Saved: {SHEET_NAME}_fig1_boxplots_all_targets.png")


# ──────────────────────────────────────────────────────────────────────────────
# 3b. FIGURE 1B — TOTAL ERROR ONLY
# ──────────────────────────────────────────────────────────────────────────────
fig1b, ax1b = plt.subplots(figsize=(15, 8.5))
fig1b.patch.set_facecolor(WHITE)
fig1b.suptitle(
    "Adjusted Position: Total Error Distribution by Target",
    fontsize=17,
    fontweight="bold",
)

draw_target_distribution_panel(ax1b, "total_error")
ax1b.set_title("Total Error (mm)", fontweight="bold", fontsize=13)

overall_total_error_mean = float(df["total_error"].mean())
ax1b.axhline(
    overall_total_error_mean,
    color="#111111",
    linestyle="-.",
    linewidth=2,
    alpha=0.85,
    zorder=2,
)
mean_label_transform = mtransforms.blended_transform_factory(ax1b.transAxes, ax1b.transData)
ax1b.text(
    -0.08,
    overall_total_error_mean + 0.02,
    f"Overall mean = {overall_total_error_mean:.2f} mm",
    transform=mean_label_transform,
    ha="right",
    va="bottom",
    fontsize=9,
    color="#111111",
    fontweight="bold",
    clip_on=False,
)

legend_handles_fig1b = legend_handles + [
    Line2D([0], [0], color="#111111", linestyle="-.", linewidth=2, label="Overall mean total error"),
]

fig1b.legend(handles=legend_handles_fig1b, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
fig1b.tight_layout(rect=[0, 0.08, 1, 0.94])
fig1b.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig1b_total_error_boxplot.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig1b)
print(f"Saved: {SHEET_NAME}_fig1b_total_error_boxplot.png")


# ──────────────────────────────────────────────────────────────────────────────
# 4. FIGURE 2 — PER-TARGET MEAN ± SD BARS
# ──────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(24, 14), sharey=True)
fig2.patch.set_facecolor(WHITE)
fig2.suptitle(
    "Adjusted Position: Per-Target Mean ± SD",
    fontsize=17,
    fontweight="bold",
)

x = np.arange(len(all_targets))
tick_rotation_all_targets, tick_align_all_targets = target_tick_rotation(len(all_targets))
fig2_upper = max(
    float((df.groupby("target_id")[metric].mean() + df.groupby("target_id")[metric].std().fillna(0)).max())
    for metric in METRICS
)
shared_fig2_ylim = compute_shared_ylim([np.array([0.0, fig2_upper], dtype=float)], include_zero=True)

for ax, metric in zip(axes2.flatten(), METRICS):
    style_axis(ax)
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

    apply_shared_y_axis(ax, shared_fig2_ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(all_targets, rotation=tick_rotation_all_targets, ha=tick_align_all_targets)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric], fontweight="bold", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
    ax.set_axisbelow(True)

fig2.tight_layout(rect=[0, 0.02, 1, 0.96])
fig2.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig2_mean_sd_bars.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig2)
print(f"Saved: {SHEET_NAME}_fig2_mean_sd_bars.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 3 — TARGET × OPERATOR HEATMAPS
# ──────────────────────────────────────────────────────────────────────────────
cmap_heat = LinearSegmentedColormap.from_list(
    "adjusted_err_cmap", ["#d8f3dc", "#ffd166", "#ef476f"]
)

fig3, axes3 = plt.subplots(2, 2, figsize=(22, 14))
fig3.patch.set_facecolor(WHITE)
fig3.suptitle(
    "Adjusted Position: Target × Operator Error Heatmaps",
    fontsize=17,
    fontweight="bold",
)

heatmap_matrices = {
    metric: (
        df.pivot_table(index="target_id", columns="operator", values=metric, aggfunc="mean")
        .reindex(index=all_targets, columns=all_operators)
    )
    for metric in METRICS
}
heatmap_values = [matrix.to_numpy(dtype=float).ravel() for matrix in heatmap_matrices.values()]
all_heatmap_values = np.concatenate(heatmap_values)
all_heatmap_values = all_heatmap_values[~np.isnan(all_heatmap_values)]
heatmap_vmin = float(np.min(all_heatmap_values)) if all_heatmap_values.size else 0.0
heatmap_vmax = float(np.max(all_heatmap_values)) if all_heatmap_values.size else 1.0

for ax, metric in zip(axes3.flatten(), METRICS):
    style_axis(ax)
    matrix = heatmap_matrices[metric]
    im = ax.imshow(matrix.values, cmap=cmap_heat, aspect="auto", vmin=heatmap_vmin, vmax=heatmap_vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"Op {op}" for op in matrix.columns], fontsize=10)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=10)
    ax.set_title(METRIC_LABELS[metric], fontweight="bold", fontsize=13)

    for (row_idx, col_idx), value in np.ndenumerate(matrix.values):
        if not np.isnan(value):
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=heatmap_label_color(float(value), heatmap_vmin, heatmap_vmax),
            )

    plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

fig3.tight_layout(rect=[0, 0.02, 1, 0.96])
fig3.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig3_operator_heatmaps.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig3)
print(f"Saved: {SHEET_NAME}_fig3_operator_heatmaps.png")


# ──────────────────────────────────────────────────────────────────────────────
# 6. FIGURE 4 — TARGET PROFILES ACROSS METRICS
# ──────────────────────────────────────────────────────────────────────────────
side_groups = [
    (side, target_mean[target_mean["side"] == side].copy())
    for side in ["Left", "Right", "Other / Midline"]
]
side_groups = [(side, frame) for side, frame in side_groups if not frame.empty]

fig4, axes4 = plt.subplots(1, len(side_groups), figsize=(7 * len(side_groups), 9), squeeze=False)
fig4.patch.set_facecolor(WHITE)
fig4.suptitle(
    "Adjusted Position: Mean Error Profile per Target (grouped by side)",
    fontsize=17,
    fontweight="bold",
)

profile_x = np.arange(len(METRICS))
profile_global_ymax = max(float(np.nanmax(target_mean[METRICS].to_numpy(dtype=float))) * 1.18, 0.5)
for ax4, (side, side_frame) in zip(axes4.flatten(), side_groups):
    color = side_color(side)
    style_axis(ax4)
    for _, row in side_frame.iterrows():
        values = row[METRICS].values.astype(float)
        target_label = str(row["target_id"])
        ax4.plot(
            profile_x,
            values,
            marker="o",
            linewidth=2,
            markersize=6,
            alpha=0.85,
            label=target_label,
        )
        ax4.text(profile_x[-1] + 0.05, values[-1], target_label, fontsize=8, va="center")

    side_mean_values = side_frame[METRICS].mean().values.astype(float)
    ax4.plot(
        profile_x,
        side_mean_values,
        color=color,
        linewidth=3.4,
        linestyle="--",
        marker="D",
        markersize=7,
        label=f"{side} mean",
        zorder=5,
    )
    for x_pos, y_val in zip(profile_x, side_mean_values):
        ax4.text(
            x_pos,
            y_val + 0.03,
            f"{y_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
            fontweight="bold",
        )
    ax4.text(
        profile_x[-1] + 0.08,
        side_mean_values[-1],
        f"{side} mean",
        fontsize=9,
        va="center",
        color=color,
        fontweight="bold",
    )

    ax4.set_xticks(profile_x)
    ax4.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax4.set_ylabel("Mean error (mm)")
    ax4.set_title(side, fontweight="bold", color=color, fontsize=13)
    ax4.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
    ax4.set_axisbelow(True)
    ax4.set_ylim(0, profile_global_ymax)
    ax4.yaxis.set_major_locator(MultipleLocator(nice_step(profile_global_ymax)))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))

fig4.tight_layout()
fig4.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig4_target_profiles.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig4)
print(f"Saved: {SHEET_NAME}_fig4_target_profiles.png\n")


# ──────────────────────────────────────────────────────────────────────────────
# 7. FIGURE 5 — RADAR PLOTS PER TARGET
# ──────────────────────────────────────────────────────────────────────────────
n_targets = len(all_targets)
n_cols = min(3, n_targets)
n_rows = int(np.ceil(n_targets / n_cols))
radar_limit = max(float(target_mean[METRICS].to_numpy(dtype=float).max()) * 1.2, 0.5)

render_target_radars(
    n_rows,
    n_cols,
    "adjusted_position_fig5_target_radars.png",
    "Adjusted Position: Radar Error Profile per Target",
)
if n_rows > n_cols:
    render_target_radars(
        n_cols,
        n_rows,
        "adjusted_position_fig5_target_radars_transposed.png",
        "Adjusted Position: Radar Error Profile per Target (Transposed Layout)",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 6 — OVERALL AVERAGE ERROR RADAR
# ──────────────────────────────────────────────────────────────────────────────
average_values = df[METRICS].mean().values.astype(float).tolist()
average_limit = max(max(average_values) * 1.35, 0.5)

fig6, ax6 = plt.subplots(figsize=(9, 8), subplot_kw={"polar": True})
fig6.patch.set_facecolor(WHITE)
fig6.suptitle(
    "Adjusted Position: Overall Average Error Radar",
    fontsize=17,
    fontweight="bold",
)

ax6.set_facecolor(WHITE)
plot_radar(ax6, average_values, color=SECONDARY_COLOR, label="Overall mean", fill_alpha=0.22)
annotate_radar_values(ax6, average_values)
ax6.set_xticks(RADAR_ANGLES[:-1])
ax6.set_xticklabels(RADAR_LABELS, fontsize=11)
ax6.set_ylim(0, average_limit)
ax6.grid(True, alpha=0.35, color=GRID_COLOR)
ax6.tick_params(axis="y", labelsize=10, colors=TEXT_COLOR)
ax6.legend(loc="upper right", bbox_to_anchor=(1.2, 1.15), frameon=False)

fig6.tight_layout()
fig6.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig6_average_error_radar.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig6)
print(f"Saved: {SHEET_NAME}_fig6_average_error_radar.png")


# ──────────────────────────────────────────────────────────────────────────────
# 9. FIGURE 7 — CORRELATION MATRIX
# ──────────────────────────────────────────────────────────────────────────────
corr = df[METRICS].corr()
short_labels = ["X", "Y", "Z", "Total"]
cmap_corr = LinearSegmentedColormap.from_list(
    "corr_cmap", ["#2E86AB", "#FFFFFF", "#E84855"]
)

fig7, ax7 = plt.subplots(figsize=(9, 8))
fig7.patch.set_facecolor(WHITE)
fig7.suptitle(
    "Adjusted Position: Pearson Correlation Matrix",
    fontsize=17,
    fontweight="bold",
)

style_axis(ax7)
im = ax7.imshow(corr.values, cmap=cmap_corr, vmin=-1, vmax=1, aspect="equal")
ax7.set_xticks(range(len(short_labels)))
ax7.set_xticklabels(short_labels, fontsize=11)
ax7.set_yticks(range(len(short_labels)))
ax7.set_yticklabels(short_labels, fontsize=11)

for (i, j), value in np.ndenumerate(corr.values):
    ax7.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=12, fontweight="bold", color=corr_label_color(value))

plt.colorbar(im, ax=ax7, shrink=0.85)
fig7.tight_layout()
fig7.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig7_correlations.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig7)
print(f"Saved: {SHEET_NAME}_fig7_correlations.png")


# ──────────────────────────────────────────────────────────────────────────────
# 10. FIGURE 8 — OPERATOR COMPARISON FOR TOTAL ERROR
# ──────────────────────────────────────────────────────────────────────────────
fig8, ax8 = plt.subplots(figsize=(12, 8))
fig8.patch.set_facecolor(WHITE)
fig8.suptitle(
    f"{SHEET_NAME}: Operator Comparison for Total Error",
    fontsize=17,
    fontweight="bold",
)

style_axis(ax8)

operator_groups = [
    df.loc[df["operator"] == operator, "total_error"].dropna().values
    for operator in all_operators
]
positions = list(range(1, len(all_operators) + 1))

box8 = ax8.boxplot(
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
for patch in box8["boxes"]:
    patch.set_facecolor(PRIMARY_COLOR)
    patch.set_alpha(0.75)

for pos, values in zip(positions, operator_groups):
    if len(values) == 0:
        continue
    jitter = rng.uniform(-0.14, 0.14, size=len(values))
    ax8.scatter(
        pos + jitter,
        values,
        color=ACCENT_COLOR,
        s=40,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

annotate_median(ax8, operator_groups, positions)
apply_fine_y_axis(ax8, operator_groups)
ax8.set_xticks(positions)
ax8.set_xticklabels([f"Operator {op}" for op in all_operators])
ax8.set_ylabel("Total Error (mm)")
ax8.set_title("Pooled across all targets", fontweight="bold", fontsize=13)

fig8.tight_layout()
fig8.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig8_operator_total_error.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig8)
print(f"Saved: {SHEET_NAME}_fig8_operator_total_error.png")


# ──────────────────────────────────────────────────────────────────────────────
# 11. FIGURE 9 — INTEROPERATOR RMSE BY TARGET
# ──────────────────────────────────────────────────────────────────────────────
rmse_source = df.merge(
    target_mean[["target_id", *METRICS]],
    on="target_id",
    how="left",
    suffixes=("", "_target_mean"),
)

interoperator_rmse_records: list[dict[str, object]] = []
global_interoperator_rmse: dict[str, object] = {"scope": "global_pooled"}

for target in ordered_targets:
    target_rows = rmse_source[rmse_source["target_id"] == target].copy()
    record: dict[str, object] = {"target_id": target, "side": target_side_map.get(target, "Other / Midline")}
    for metric in METRICS:
        diff = target_rows[metric] - target_rows[f"{metric}_target_mean"]
        record[metric] = rmse(diff)
    interoperator_rmse_records.append(record)

for metric in METRICS:
    diff = rmse_source[metric] - rmse_source[f"{metric}_target_mean"]
    global_interoperator_rmse[metric] = rmse(diff)

interoperator_rmse_by_target = pd.DataFrame(interoperator_rmse_records)
interoperator_rmse_global = pd.DataFrame([global_interoperator_rmse])

interoperator_rmse_by_target.to_csv(OUTPUT_DIR / f"{SHEET_NAME}_interoperator_rmse_by_target.csv", index=False)
interoperator_rmse_global.to_csv(OUTPUT_DIR / f"{SHEET_NAME}_interoperator_rmse_global.csv", index=False)

print("\n--- Interoperator RMSE by target ---")
print(interoperator_rmse_by_target.round(3).to_string(index=False))
print("\n--- Global pooled interoperator RMSE ---")
print(interoperator_rmse_global.round(3).to_string(index=False))
print(f"Saved: {SHEET_NAME}_interoperator_rmse_by_target.csv")
print(f"Saved: {SHEET_NAME}_interoperator_rmse_global.csv")

fig9, ax9 = plt.subplots(figsize=(14, 8.5))
fig9.patch.set_facecolor(WHITE)
fig9.suptitle(
    f"{SHEET_NAME}: Interoperator RMSE by Target",
    fontsize=17,
    fontweight="bold",
)

style_axis(ax9)

x9 = np.arange(len(ordered_targets))
bar_width = 0.18
offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(METRICS))
tick_rotation_targets, tick_align_targets = target_tick_rotation(len(ordered_targets))

for offset, metric in zip(offsets, METRICS):
    values = interoperator_rmse_by_target[metric].astype(float).values
    bars = ax9.bar(
        x9 + offset,
        values,
        width=bar_width,
        color=METRIC_COLORS[metric],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
        label=METRIC_LABELS[metric],
    )
    global_rmse_value = float(interoperator_rmse_global.loc[0, metric])
    ax9.axhline(
        global_rmse_value,
        color=METRIC_COLORS[metric],
        linestyle=(0, (4, 3)),
        linewidth=1.8,
        alpha=0.85,
        zorder=2,
    )
    for bar, value in zip(bars, values):
        ax9.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax9.set_xticks(x9)
ax9.set_xticklabels(ordered_targets, rotation=tick_rotation_targets, ha=tick_align_targets)
ax9.set_ylabel("RMSE (mm)")
ax9.set_title("Per-target operator disagreement; dashed lines show global pooled RMSE for each metric", fontweight="bold", fontsize=13)
ax9.yaxis.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
ax9.set_axisbelow(True)
ax9.legend(ncol=2, frameon=False)
rmse_ymax = max(
    float(np.nanmax(interoperator_rmse_by_target[METRICS].to_numpy(dtype=float))),
    float(np.nanmax(interoperator_rmse_global[METRICS].to_numpy(dtype=float))),
) * 1.18
rmse_ymax = max(rmse_ymax, 0.5)
apply_shared_y_axis(ax9, (0.0, rmse_ymax))

fig9.tight_layout()
fig9.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig9_operator_rmse.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig9)
print(f"Saved: {SHEET_NAME}_fig9_operator_rmse.png")


# ──────────────────────────────────────────────────────────────────────────────
# 12. FIGURE 10 — EFFECT OF TRAJECTORY ORIENTATION
# ──────────────────────────────────────────────────────────────────────────────
collar_effect_df = df[df["arc"] == 100].copy()
arc_effect_df = df[df["collar"] == 75].copy()


def summarize_orientation_effect(source_df: pd.DataFrame, angle_col: str) -> tuple[pd.DataFrame, float, float, float, float]:
    summary = (
        source_df.groupby(angle_col)["total_error"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(angle_col)
    )
    groups = [group["total_error"].dropna().to_numpy(dtype=float) for _, group in source_df.groupby(angle_col)]
    f_stat, p_value = permutation_anova_pvalue(groups)
    overall_mean = float(source_df["total_error"].mean())
    overall_std = float(source_df["total_error"].std(ddof=1))
    return summary, overall_mean, overall_std, f_stat, p_value


collar_summary, collar_mean, collar_std, collar_f, collar_p = summarize_orientation_effect(collar_effect_df, "collar")
arc_summary, arc_mean, arc_std, arc_f, arc_p = summarize_orientation_effect(arc_effect_df, "arc")

print("\n--- Trajectory orientation effect on Euclidean error ---")
print(f"Collar angles (arc fixed at 100°): {collar_mean:.3f} ± {collar_std:.3f} mm | permutation ANOVA F = {collar_f:.3f}, p = {collar_p:.4f}")
print(collar_summary.round(3).to_string(index=False))
print(f"Arc angles (collar fixed at 75°):  {arc_mean:.3f} ± {arc_std:.3f} mm | permutation ANOVA F = {arc_f:.3f}, p = {arc_p:.4f}")
print(arc_summary.round(3).to_string(index=False))


def plot_orientation_panel(
    ax: plt.Axes,
    source_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    angle_col: str,
    title: str,
    subtitle: str,
    color: str,
    y_limits: tuple[float, float],
) -> None:
    angle_values = summary_df[angle_col].tolist()
    positions = np.arange(len(angle_values))
    angle_to_position = dict(zip(angle_values, positions))
    style_axis(ax)

    for angle in angle_values:
        values = source_df.loc[source_df[angle_col] == angle, "total_error"].dropna().to_numpy(dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), angle_to_position[angle]) + jitter,
            values,
            s=42,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

    mean_values = summary_df["mean"].to_numpy(dtype=float)
    std_values = summary_df["std"].fillna(0).to_numpy(dtype=float)
    ax.errorbar(
        positions,
        mean_values,
        yerr=std_values,
        fmt="-D",
        color="#111111",
        ecolor="#111111",
        elinewidth=1.4,
        capsize=5,
        markersize=6,
        linewidth=2.2,
        zorder=4,
    )

    for x_pos, row in zip(positions, summary_df.itertuples(index=False)):
        std_val = 0.0 if pd.isna(row.std) else float(row.std)
        ax.text(
            x_pos,
            row.mean + std_val + 0.04,
            f"{row.mean:.2f} ± {std_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#111111",
            fontweight="bold",
        )

    apply_shared_y_axis(ax, y_limits)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{angle}°" for angle in angle_values], fontsize=10)
    ax.set_ylabel("Euclidean error (mm)")
    ax.set_title(f"{title}\n{subtitle}", fontweight="bold", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
    ax.set_axisbelow(True)

    y_lo2, _ = ax.get_ylim()
    for x_pos, row in zip(positions, summary_df.itertuples(index=False)):
        ax.text(x_pos, y_lo2, f"n={int(row.count)}", ha="center", va="bottom", fontsize=8, color="gray")


fig10, axes10 = plt.subplots(1, 2, figsize=(18, 8.5), sharey=True)
fig10.patch.set_facecolor(WHITE)
fig10.suptitle(
    "Adjusted Position: Effect of Trajectory Orientation on Euclidean Error",
    fontsize=17,
    fontweight="bold",
)

orientation_shared_ylim = compute_shared_ylim(
    [
        collar_effect_df["total_error"].dropna().to_numpy(dtype=float),
        arc_effect_df["total_error"].dropna().to_numpy(dtype=float),
    ],
    include_zero=True,
)

plot_orientation_panel(
    axes10[0],
    collar_effect_df,
    collar_summary,
    "collar",
    "Collar angle effect",
    f"Arc fixed at 100° | overall {collar_mean:.2f} ± {collar_std:.2f} mm | p = {collar_p:.3f}",
    PRIMARY_COLOR,
    orientation_shared_ylim,
)
plot_orientation_panel(
    axes10[1],
    arc_effect_df,
    arc_summary,
    "arc",
    "Arc angle effect",
    f"Collar fixed at 75° | overall {arc_mean:.2f} ± {arc_std:.2f} mm | p = {arc_p:.3f}",
    ACCENT_COLOR,
    orientation_shared_ylim,
)

legend10 = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=7, label="Individual examiner values"),
    Line2D([0], [0], marker="D", color="#111111", markerfacecolor="#111111", markersize=6, linewidth=2, label="Mean ± SD"),
]
fig10.legend(handles=legend10, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))

fig10.tight_layout(rect=[0, 0.06, 1, 0.94])
fig10.savefig(OUTPUT_DIR / f"{SHEET_NAME}_fig10_orientation_effects.png", bbox_inches="tight", facecolor=WHITE)
plt.close(fig10)
print(f"Saved: {SHEET_NAME}_fig10_orientation_effects.png")

print(f"\n✔ {SHEET_NAME} analysis complete.")
