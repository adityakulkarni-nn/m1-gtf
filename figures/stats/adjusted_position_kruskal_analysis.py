from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "master_error.xlsx"
DEFAULT_SHEET = "adjusted_position"
DEFAULT_GROUP_COLS = ["side", "collar", "arc", "operator"]
DEFAULT_OUTCOME = "total_error"
DEFAULT_ALPHA = 0.05
DEFAULT_OUTPUT_XLSX = SCRIPT_DIR / "adjusted_position_kruskal_results.xlsx"
DEFAULT_OUTPUT_MD = SCRIPT_DIR / "adjusted_position_kruskal_significance.md"


@dataclass(frozen=True)
class AnalysisConfig:
    input_file: Path
    sheet_name: str
    outcome_col: str
    group_cols: list[str]
    alpha: float
    output_excel: Path
    output_markdown: Path


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run Shapiro-Wilk normality checks and Kruskal-Wallis analysis for "
            "total_error on the adjusted_position sheet of master_error.xlsx."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the Excel workbook.")
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help="Sheet name to analyze. 'adjusted' is accepted as an alias for 'adjusted_position'.",
    )
    parser.add_argument("--outcome", default=DEFAULT_OUTCOME, help="Outcome column to analyze.")
    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=DEFAULT_GROUP_COLS,
        help="Grouping columns to test with Kruskal-Wallis.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Significance level.")
    parser.add_argument(
        "--output-excel",
        type=Path,
        default=DEFAULT_OUTPUT_XLSX,
        help="Excel workbook for detailed results.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Markdown file for narrative significance interpretation.",
    )
    args = parser.parse_args()

    sheet_name = args.sheet
    if sheet_name.strip().lower() == "adjusted":
        sheet_name = DEFAULT_SHEET

    return AnalysisConfig(
        input_file=args.input,
        sheet_name=sheet_name,
        outcome_col=args.outcome,
        group_cols=list(args.group_cols),
        alpha=float(args.alpha),
        output_excel=args.output_excel,
        output_markdown=args.output_markdown,
    )


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing {label} columns: {missing}")


def load_data(input_file: Path, sheet_name: str) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Workbook not found: {input_file}")

    workbook = pd.ExcelFile(input_file)
    if sheet_name not in workbook.sheet_names:
        raise KeyError(f"Sheet '{sheet_name}' not found. Available sheets: {workbook.sheet_names}")

    df = pd.read_excel(input_file, sheet_name=sheet_name).copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_group_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def shapiro_record(values: np.ndarray, alpha: float) -> dict[str, object]:
    record: dict[str, object] = {
        "n": int(values.size),
        "w_stat": np.nan,
        "p_value": np.nan,
        "normal_at_alpha": np.nan,
        "test_ran": False,
        "notes": "",
    }
    if values.size < 3:
        record["notes"] = "Shapiro-Wilk requires at least 3 observations."
        return record

    w_stat, p_value = stats.shapiro(values)
    record.update(
        {
            "w_stat": float(w_stat),
            "p_value": float(p_value),
            "normal_at_alpha": bool(p_value >= alpha),
            "test_ran": True,
        }
    )
    return record


def compute_normality_results(
    df: pd.DataFrame,
    outcome_col: str,
    group_cols: list[str],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome_values = to_numeric(df[outcome_col]).dropna().to_numpy(dtype=float)
    overall_record = {
        "scope": "overall",
        "factor": "ALL",
        "level": "ALL",
        **shapiro_record(outcome_values, alpha),
    }
    overall_df = pd.DataFrame([overall_record])

    records: list[dict[str, object]] = []
    for factor in group_cols:
        subset = df[[factor, outcome_col]].copy()
        subset[outcome_col] = to_numeric(subset[outcome_col])
        subset = subset.dropna(subset=[factor, outcome_col])
        for level, group_df in subset.groupby(factor, dropna=True, sort=True):
            values = group_df[outcome_col].dropna().to_numpy(dtype=float)
            records.append(
                {
                    "scope": "group",
                    "factor": factor,
                    "level": format_group_value(level),
                    **shapiro_record(values, alpha),
                }
            )

    by_factor_df = pd.DataFrame.from_records(records)
    if by_factor_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "factor",
                "groups_evaluated",
                "groups_tested",
                "groups_non_normal",
                "groups_insufficient_n",
                "all_tested_groups_normal",
            ]
        )
        return overall_df, by_factor_df, summary_df

    summary_rows: list[dict[str, object]] = []
    for factor, factor_df in by_factor_df.groupby("factor", sort=True):
        tested = factor_df["test_ran"] == True
        normal = factor_df["normal_at_alpha"] == True
        non_normal = factor_df["normal_at_alpha"] == False
        summary_rows.append(
            {
                "factor": factor,
                "groups_evaluated": int(len(factor_df)),
                "groups_tested": int(tested.sum()),
                "groups_non_normal": int((tested & non_normal).sum()),
                "groups_insufficient_n": int((factor_df["test_ran"] == False).sum()),
                "all_tested_groups_normal": bool(factor_df.loc[tested, "normal_at_alpha"].all()) if tested.any() else True,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return overall_df, by_factor_df, summary_df


def epsilon_squared_kruskal(h_stat: float, n_total: int, n_groups: int) -> float:
    denominator = n_total - n_groups
    if denominator <= 0:
        return float("nan")
    return max(0.0, float((h_stat - n_groups + 1) / denominator))


def effect_size_label(epsilon_sq: float) -> str:
    if pd.isna(epsilon_sq):
        return "undetermined"
    if epsilon_sq < 0.01:
        return "negligible"
    if epsilon_sq < 0.08:
        return "small"
    if epsilon_sq < 0.26:
        return "medium"
    return "large"


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed, start=1):
        holm_value = (m - rank + 1) * float(p_value)
        running_max = max(running_max, holm_value)
        adjusted[original_idx] = min(1.0, running_max)
    return adjusted


def run_kruskal_for_factor(
    df: pd.DataFrame,
    factor: str,
    outcome_col: str,
    alpha: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    subset = df[[factor, outcome_col]].copy()
    subset[outcome_col] = to_numeric(subset[outcome_col])
    subset = subset.dropna(subset=[factor, outcome_col])

    grouped_values: list[np.ndarray] = []
    group_order: list[str] = []
    group_sizes: dict[str, int] = {}
    group_medians: dict[str, float] = {}

    for level, group_df in subset.groupby(factor, dropna=True, sort=True):
        values = group_df[outcome_col].to_numpy(dtype=float)
        if values.size == 0:
            continue
        group_name = format_group_value(level)
        grouped_values.append(values)
        group_order.append(group_name)
        group_sizes[group_name] = int(values.size)
        group_medians[group_name] = float(np.median(values))

    result_row: dict[str, object] = {
        "factor": factor,
        "outcome": outcome_col,
        "alpha": alpha,
        "n_groups": len(grouped_values),
        "df": max(0, len(grouped_values) - 1),
        "total_n": int(sum(group_sizes.values())),
        "group_sizes": "; ".join(f"{level}={size}" for level, size in group_sizes.items()),
        "group_medians": "; ".join(f"{level}={group_medians[level]:.4f}" for level in group_order),
        "h_stat": np.nan,
        "p_value": np.nan,
        "epsilon_squared": np.nan,
        "effect_size": "undetermined",
        "significant": False,
        "pairwise_run": False,
        "notes": "",
    }

    if len(grouped_values) < 2:
        result_row["notes"] = "At least 2 non-empty groups are required for Kruskal-Wallis."
        return result_row, pd.DataFrame()

    h_stat, p_value = stats.kruskal(*grouped_values)
    epsilon_sq = epsilon_squared_kruskal(float(h_stat), int(sum(group_sizes.values())), len(grouped_values))
    result_row.update(
        {
            "h_stat": float(h_stat),
            "p_value": float(p_value),
            "epsilon_squared": epsilon_sq,
            "effect_size": effect_size_label(epsilon_sq),
            "significant": bool(p_value < alpha),
        }
    )

    if p_value >= alpha:
        return result_row, pd.DataFrame()

    pairwise_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    pair_keys: list[tuple[str, str]] = []

    for i, group_a in enumerate(group_order[:-1]):
        values_a = subset.loc[subset[factor].map(format_group_value) == group_a, outcome_col].to_numpy(dtype=float)
        for group_b in group_order[i + 1:]:
            values_b = subset.loc[subset[factor].map(format_group_value) == group_b, outcome_col].to_numpy(dtype=float)
            u_stat, raw_p = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            pair_keys.append((group_a, group_b))
            raw_p_values.append(float(raw_p))
            pairwise_rows.append(
                {
                    "factor": factor,
                    "outcome": outcome_col,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": int(values_a.size),
                    "n_b": int(values_b.size),
                    "median_a": float(np.median(values_a)),
                    "median_b": float(np.median(values_b)),
                    "u_stat": float(u_stat),
                    "raw_p": float(raw_p),
                }
            )

    adjusted_p_values = holm_adjust(raw_p_values)
    for row, holm_p in zip(pairwise_rows, adjusted_p_values):
        row["holm_p"] = float(holm_p)
        row["significant"] = bool(holm_p < alpha)

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values(by=["holm_p", "raw_p", "group_a", "group_b"]).reset_index(drop=True)
        result_row["pairwise_run"] = True

    return result_row, pairwise_df


def build_significant_interpretation(row: pd.Series, pairwise_df: pd.DataFrame, alpha: float) -> str:
    medians_text = row["group_medians"] if pd.notna(row["group_medians"]) else ""
    intro = (
        f"Kruskal-Wallis showed a significant difference in {row['outcome']} across {row['factor']} "
        f"(H({int(row['df'])}) = {row['h_stat']:.3f}, p = {row['p_value']:.4f}, "
        f"epsilon-squared = {row['epsilon_squared']:.3f} [{row['effect_size']}])."
    )
    if medians_text:
        intro += f" Group medians were: {medians_text}."

    if pairwise_df.empty:
        return intro + " No pairwise comparisons were available."

    significant_pairs = pairwise_df[pairwise_df["significant"] == True].copy()
    if significant_pairs.empty:
        return (
            intro
            + " No Holm-corrected pairwise Mann-Whitney comparison remained significant, "
            + "so the omnibus effect should be interpreted as an overall distributional difference."
        )

    direction_bits: list[str] = []
    for _, pair in significant_pairs.iterrows():
        if pair["median_a"] > pair["median_b"]:
            higher_group, lower_group = pair["group_a"], pair["group_b"]
            higher_median, lower_median = pair["median_a"], pair["median_b"]
        elif pair["median_b"] > pair["median_a"]:
            higher_group, lower_group = pair["group_b"], pair["group_a"]
            higher_median, lower_median = pair["median_b"], pair["median_a"]
        else:
            higher_group, lower_group = pair["group_a"], pair["group_b"]
            higher_median, lower_median = pair["median_a"], pair["median_b"]

        direction_bits.append(
            f"{higher_group} was higher than {lower_group} "
            f"(Holm-adjusted p = {pair['holm_p']:.4f}; medians {higher_group} = {higher_median:.3f}, "
            f"{lower_group} = {lower_median:.3f})"
        )

    return intro + " Post hoc comparisons showed: " + "; ".join(direction_bits) + "."


def build_markdown(
    config: AnalysisConfig,
    df: pd.DataFrame,
    overall_normality_df: pd.DataFrame,
    normality_summary_df: pd.DataFrame,
    kruskal_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
) -> str:
    overall = overall_normality_df.iloc[0]
    lines: list[str] = []
    lines.append("# Adjusted Sheet Total Error Kruskal-Wallis Analysis")
    lines.append("")
    lines.append(f"- Workbook: {config.input_file.name}")
    lines.append(f"- Sheet analyzed: {config.sheet_name}")
    lines.append(f"- Outcome analyzed: {config.outcome_col}")
    lines.append(f"- Rows analyzed: {len(df)}")
    lines.append(f"- Factors analyzed: {', '.join(config.group_cols)}")
    lines.append(f"- Significance threshold: alpha = {config.alpha}")
    lines.append("")
    lines.append("## Normality check")
    lines.append("")
    if bool(overall["test_ran"]):
        normality_label = "approximately normal" if bool(overall["normal_at_alpha"]) else "non-normal"
        lines.append(
            f"Overall `total_error` distribution was {normality_label} by Shapiro-Wilk "
            f"(W = {overall['w_stat']:.3f}, p = {overall['p_value']:.4f}, n = {int(overall['n'])})."
        )
    else:
        lines.append(f"Overall normality test was not run: {overall['notes']}")

    if not normality_summary_df.empty:
        lines.append("")
        lines.append("### Group-level normality summary")
        lines.append("")
        lines.append("| Factor | Groups evaluated | Groups tested | Non-normal groups | Insufficient n | All tested groups normal |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for _, row in normality_summary_df.sort_values("factor").iterrows():
            lines.append(
                "| "
                + f"{row['factor']} | {int(row['groups_evaluated'])} | {int(row['groups_tested'])} | "
                + f"{int(row['groups_non_normal'])} | {int(row['groups_insufficient_n'])} | "
                + f"{'Yes' if bool(row['all_tested_groups_normal']) else 'No'} |"
            )

    lines.append("")
    lines.append("## Significant omnibus findings")
    lines.append("")
    significant_df = kruskal_df[kruskal_df["significant"] == True].copy()
    if significant_df.empty:
        lines.append("No factor showed a significant Kruskal-Wallis difference in `total_error` at alpha = 0.05.")
    else:
        lines.append("| Factor | Outcome | H | df | p-value | Epsilon-squared | Effect size |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for _, row in significant_df.sort_values(["p_value", "factor"]).iterrows():
            lines.append(
                f"| {row['factor']} | {row['outcome']} | {row['h_stat']:.3f} | {int(row['df'])} | "
                f"{row['p_value']:.4f} | {row['epsilon_squared']:.3f} | {row['effect_size']} |"
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if interpretation_df.empty:
        lines.append(
            "Because overall `total_error` normality was not supported and no grouping factor reached "
            "Kruskal-Wallis significance, the current adjusted-position data do not show evidence that "
            "`total_error` differs across the tested columns at alpha = 0.05."
        )
    else:
        for _, row in interpretation_df.iterrows():
            factor = row["factor"]
            lines.append(f"### {factor} × {config.outcome_col}")
            lines.append("")
            lines.append(str(row["interpretation"]))
            lines.append("")
            factor_pairs = pairwise_df[pairwise_df["factor"] == factor].copy()
            if factor_pairs.empty:
                continue
            lines.append("Pairwise Mann-Whitney comparisons:")
            lines.append("")
            lines.append("| Group A | Group B | Median A | Median B | Raw p | Holm p | Significant |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
            for _, pair in factor_pairs.iterrows():
                lines.append(
                    f"| {pair['group_a']} | {pair['group_b']} | {pair['median_a']:.3f} | {pair['median_b']:.3f} | "
                    f"{pair['raw_p']:.4f} | {pair['holm_p']:.4f} | {'Yes' if bool(pair['significant']) else 'No'} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_excel_results(
    config: AnalysisConfig,
    overall_normality_df: pd.DataFrame,
    by_factor_normality_df: pd.DataFrame,
    normality_summary_df: pd.DataFrame,
    kruskal_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
) -> None:
    metadata_df = pd.DataFrame(
        [
            {"field": "workbook", "value": str(config.input_file)},
            {"field": "sheet", "value": config.sheet_name},
            {"field": "outcome", "value": config.outcome_col},
            {"field": "alpha", "value": config.alpha},
            {"field": "group_cols", "value": ", ".join(config.group_cols)},
        ]
    )

    config.output_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.output_excel, engine="openpyxl") as writer:
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)
        overall_normality_df.to_excel(writer, sheet_name="normality_overall", index=False)
        by_factor_normality_df.to_excel(writer, sheet_name="normality_by_factor", index=False)
        normality_summary_df.to_excel(writer, sheet_name="normality_summary", index=False)
        kruskal_df.to_excel(writer, sheet_name="kruskal_results", index=False)
        pairwise_df.to_excel(writer, sheet_name="pairwise_results", index=False)
        interpretation_df.to_excel(writer, sheet_name="interpretations", index=False)


def print_console_summary(
    config: AnalysisConfig,
    df: pd.DataFrame,
    overall_normality_df: pd.DataFrame,
    kruskal_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
) -> None:
    overall = overall_normality_df.iloc[0]
    print("=" * 78)
    print("ADJUSTED SHEET TOTAL ERROR KRUSKAL-WALLIS ANALYSIS")
    print("=" * 78)
    print(f"Workbook           : {config.input_file}")
    print(f"Sheet              : {config.sheet_name}")
    print(f"Outcome            : {config.outcome_col}")
    print(f"Rows analyzed      : {len(df)}")
    print(f"Factors analyzed   : {', '.join(config.group_cols)}")
    print(f"Alpha              : {config.alpha}")
    print()
    if bool(overall["test_ran"]):
        print(
            f"Overall normality  : W = {overall['w_stat']:.3f}, p = {overall['p_value']:.4f} "
            f"-> {'normal' if bool(overall['normal_at_alpha']) else 'non-normal'}"
        )
    else:
        print(f"Overall normality  : not tested ({overall['notes']})")

    print("\nKruskal-Wallis results:")
    display_cols = ["factor", "n_groups", "h_stat", "p_value", "epsilon_squared", "effect_size", "significant"]
    print(kruskal_df[display_cols].to_string(index=False))

    print("\nSignificant interpretations:")
    if interpretation_df.empty:
        print("  None. No factor reached alpha = 0.05 for total_error.")
    else:
        for _, row in interpretation_df.iterrows():
            print(f"- {row['factor']}: {row['interpretation']}")

    print(f"\nExcel results      : {config.output_excel}")
    print(f"Markdown summary   : {config.output_markdown}")


def main() -> None:
    config = parse_args()
    df = load_data(config.input_file, config.sheet_name)
    require_columns(df, [config.outcome_col, *config.group_cols], "analysis")

    overall_normality_df, by_factor_normality_df, normality_summary_df = compute_normality_results(
        df=df,
        outcome_col=config.outcome_col,
        group_cols=config.group_cols,
        alpha=config.alpha,
    )

    kruskal_rows: list[dict[str, object]] = []
    pairwise_frames: list[pd.DataFrame] = []
    interpretation_rows: list[dict[str, object]] = []

    for factor in config.group_cols:
        result_row, pairwise_df = run_kruskal_for_factor(
            df=df,
            factor=factor,
            outcome_col=config.outcome_col,
            alpha=config.alpha,
        )
        kruskal_rows.append(result_row)
        if not pairwise_df.empty:
            pairwise_frames.append(pairwise_df)
        if result_row["significant"]:
            interpretation_rows.append(
                {
                    "factor": factor,
                    "outcome": config.outcome_col,
                    "interpretation": build_significant_interpretation(
                        pd.Series(result_row),
                        pairwise_df,
                        config.alpha,
                    ),
                }
            )

    kruskal_df = pd.DataFrame(kruskal_rows)
    pairwise_df = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame(
        columns=["factor", "outcome", "group_a", "group_b", "n_a", "n_b", "median_a", "median_b", "u_stat", "raw_p", "holm_p", "significant"]
    )
    interpretation_df = pd.DataFrame(interpretation_rows)

    markdown_text = build_markdown(
        config=config,
        df=df,
        overall_normality_df=overall_normality_df,
        normality_summary_df=normality_summary_df,
        kruskal_df=kruskal_df,
        pairwise_df=pairwise_df,
        interpretation_df=interpretation_df,
    )

    write_excel_results(
        config=config,
        overall_normality_df=overall_normality_df,
        by_factor_normality_df=by_factor_normality_df,
        normality_summary_df=normality_summary_df,
        kruskal_df=kruskal_df,
        pairwise_df=pairwise_df,
        interpretation_df=interpretation_df,
    )
    config.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    config.output_markdown.write_text(markdown_text, encoding="utf-8")
    print_console_summary(
        config=config,
        df=df,
        overall_normality_df=overall_normality_df,
        kruskal_df=kruskal_df,
        interpretation_df=interpretation_df,
    )


if __name__ == "__main__":
    main()