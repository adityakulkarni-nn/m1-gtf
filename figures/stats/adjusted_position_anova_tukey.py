"""
Run Shapiro–Wilk normality checks, one-way ANOVA, and post hoc Tukey HSD
comparisons on the `adjusted_position` sheet of `master_error.xlsx`.

Default analyses
----------------
Grouping factors:
- target_id
- side
- collar
- arc
- operator

Outcome columns:
- error_x
- error_y
- diff_distance
- error_z
- total_error

Outputs
-------
Written next to this script by default:
- adjusted_position_parametric_analysis.xlsx
- adjusted_position_shapiro_results.csv
- adjusted_position_anova_results.csv
- adjusted_position_tukey_results.csv

Usage
-----
python adjusted_position_anova_tukey.py
python adjusted_position_anova_tukey.py --group-cols operator arc --value-cols total_error error_z
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "master_error.xlsx"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_SHEET = "adjusted_position"
DEFAULT_GROUP_COLS = ["target_id", "side", "collar", "arc", "operator"]
DEFAULT_VALUE_COLS = ["error_x", "error_y", "diff_distance", "error_z", "total_error"]
DEFAULT_ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess normality with Shapiro–Wilk, run one-way ANOVA, and apply "
            "Tukey HSD when ANOVA p < alpha."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the Excel workbook.")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help="Sheet name to analyze.")
    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=DEFAULT_GROUP_COLS,
        help="Grouping columns to use as one-way ANOVA factors.",
    )
    parser.add_argument(
        "--value-cols",
        nargs="+",
        default=DEFAULT_VALUE_COLS,
        help="Numeric outcome columns to analyze.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Significance level.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Excel and CSV outputs.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing {label} columns: {missing}")


def load_data(input_file: Path, sheet_name: str) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Workbook not found: {input_file}")

    df = pd.read_excel(input_file, sheet_name=sheet_name).copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def shapiro_results_for_factor(
    df: pd.DataFrame,
    factor: str,
    value_col: str,
    alpha: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    subset = df[[factor, value_col]].copy()
    subset[value_col] = to_numeric_series(subset[value_col])
    subset = subset.dropna(subset=[value_col])

    overall_values = subset[value_col].dropna().to_numpy(dtype=float)
    overall_record = {
        "factor": factor,
        "outcome": value_col,
        "scope": "overall",
        "level": "ALL",
        "n": int(overall_values.size),
        "w_stat": np.nan,
        "p_value": np.nan,
        "normal_at_alpha": np.nan,
        "test_ran": False,
        "notes": "",
    }
    if overall_values.size >= 3:
        w_stat, p_value = stats.shapiro(overall_values)
        overall_record.update(
            {
                "w_stat": float(w_stat),
                "p_value": float(p_value),
                "normal_at_alpha": bool(p_value >= alpha),
                "test_ran": True,
            }
        )
    else:
        overall_record["notes"] = "Shapiro–Wilk requires at least 3 observations."
    records.append(overall_record)

    for level, group_df in subset.groupby(factor, dropna=True, sort=True):
        values = group_df[value_col].dropna().to_numpy(dtype=float)
        record = {
            "factor": factor,
            "outcome": value_col,
            "scope": "group",
            "level": str(level),
            "n": int(values.size),
            "w_stat": np.nan,
            "p_value": np.nan,
            "normal_at_alpha": np.nan,
            "test_ran": False,
            "notes": "",
        }
        if values.size >= 3:
            w_stat, p_value = stats.shapiro(values)
            record.update(
                {
                    "w_stat": float(w_stat),
                    "p_value": float(p_value),
                    "normal_at_alpha": bool(p_value >= alpha),
                    "test_ran": True,
                }
            )
        else:
            record["notes"] = "Shapiro–Wilk requires at least 3 observations."
        records.append(record)

    return pd.DataFrame.from_records(records)


def anova_and_tukey_for_factor(
    df: pd.DataFrame,
    factor: str,
    value_col: str,
    alpha: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    subset = df[[factor, value_col]].copy()
    subset[value_col] = to_numeric_series(subset[value_col])
    subset = subset.dropna(subset=[factor, value_col])

    groups: list[np.ndarray] = []
    group_sizes: dict[str, int] = {}
    for level, group_df in subset.groupby(factor, dropna=True, sort=True):
        values = group_df[value_col].to_numpy(dtype=float)
        if values.size == 0:
            continue
        groups.append(values)
        group_sizes[str(level)] = int(values.size)

    result_row: dict[str, object] = {
        "factor": factor,
        "outcome": value_col,
        "alpha": alpha,
        "n_groups": len(groups),
        "total_n": int(sum(group_sizes.values())),
        "group_sizes": "; ".join(f"{level}={size}" for level, size in group_sizes.items()),
        "f_stat": np.nan,
        "p_value": np.nan,
        "significant": False,
        "tukey_run": False,
        "notes": "",
    }

    if len(groups) < 2:
        result_row["notes"] = "At least 2 non-empty groups are required for ANOVA."
        return result_row, pd.DataFrame()

    f_stat, p_value = stats.f_oneway(*groups)
    result_row.update(
        {
            "f_stat": float(f_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
        }
    )

    if p_value >= alpha:
        return result_row, pd.DataFrame()

    tukey = pairwise_tukeyhsd(
        endog=subset[value_col].to_numpy(dtype=float),
        groups=subset[factor].astype(str).to_numpy(),
        alpha=alpha,
    )
    summary_rows = tukey.summary().data
    tukey_df = pd.DataFrame(summary_rows[1:], columns=summary_rows[0])
    tukey_df = tukey_df.rename(columns={"p-adj": "p_adj"})
    for numeric_col in ["meandiff", "p_adj", "lower", "upper"]:
        if numeric_col in tukey_df.columns:
            tukey_df[numeric_col] = pd.to_numeric(tukey_df[numeric_col], errors="coerce")
    if "reject" in tukey_df.columns:
        tukey_df["reject"] = tukey_df["reject"].astype(str).str.lower().eq("true")

    tukey_df.insert(0, "outcome", value_col)
    tukey_df.insert(0, "factor", factor)
    result_row["tukey_run"] = True
    return result_row, tukey_df


def summarize_normality(shapiro_df: pd.DataFrame) -> pd.DataFrame:
    group_rows = shapiro_df[shapiro_df["scope"] == "group"].copy()
    if group_rows.empty:
        return pd.DataFrame(columns=[
            "factor",
            "outcome",
            "groups_evaluated",
            "groups_tested",
            "groups_normal",
            "groups_non_normal",
            "groups_insufficient_n",
            "all_tested_groups_normal",
        ])

    def build_summary(group: pd.DataFrame) -> pd.Series:
        tested_rows = group[group["test_ran"] == True]
        tested_normal = tested_rows["normal_at_alpha"].astype(bool) if not tested_rows.empty else pd.Series(dtype=bool)
        return pd.Series(
            {
                "groups_evaluated": int(len(group)),
                "groups_tested": int(group["test_ran"].sum()),
                "groups_normal": int(((group["test_ran"] == True) & (group["normal_at_alpha"] == True)).sum()),
                "groups_non_normal": int(((group["test_ran"] == True) & (group["normal_at_alpha"] == False)).sum()),
                "groups_insufficient_n": int((group["test_ran"] == False).sum()),
                "all_tested_groups_normal": bool(tested_normal.all()) if not tested_rows.empty else True,
            }
        )

    summary = (
        group_rows.groupby(["factor", "outcome"], as_index=False)
        .apply(build_summary, include_groups=False)
        .reset_index(drop=True)
    )
    return summary


def print_console_summary(
    df: pd.DataFrame,
    input_file: Path,
    sheet_name: str,
    group_cols: list[str],
    value_cols: list[str],
    shapiro_df: pd.DataFrame,
    anova_df: pd.DataFrame,
    tukey_df: pd.DataFrame,
) -> None:
    print("=" * 78)
    print("PARAMETRIC ANALYSIS SUMMARY")
    print("=" * 78)
    print(f"Workbook           : {input_file}")
    print(f"Sheet              : {sheet_name}")
    print(f"Rows analyzed      : {len(df)}")
    print(f"Factors analyzed   : {', '.join(group_cols)}")
    print(f"Outcomes analyzed  : {', '.join(value_cols)}")

    normality_summary = summarize_normality(shapiro_df)
    if not normality_summary.empty:
        print("\nShapiro–Wilk by factor / outcome:")
        print(normality_summary.to_string(index=False))

    print("\nANOVA results:")
    display_cols = ["factor", "outcome", "n_groups", "total_n", "f_stat", "p_value", "significant"]
    print(anova_df[display_cols].round({"f_stat": 4, "p_value": 6}).to_string(index=False))

    significant = anova_df[anova_df["significant"] == True]
    if significant.empty:
        print("\nNo ANOVA result met the significance threshold, so no Tukey tests were produced.")
        return

    print("\nSignificant ANOVA results (Tukey HSD run):")
    print(significant[["factor", "outcome", "f_stat", "p_value"]].round({"f_stat": 4, "p_value": 6}).to_string(index=False))
    if not tukey_df.empty:
        print("\nTukey HSD results:")
        print(tukey_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    df = load_data(args.input, args.sheet)

    require_columns(df, args.group_cols, "group")
    require_columns(df, args.value_cols, "value")

    available_group_cols = [
        column for column in args.group_cols
        if df[column].dropna().nunique() > 1
    ]
    skipped_group_cols = sorted(set(args.group_cols) - set(available_group_cols))
    if not available_group_cols:
        raise ValueError("No grouping columns with at least 2 levels were available for ANOVA.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shapiro_frames: list[pd.DataFrame] = []
    anova_rows: list[dict[str, object]] = []
    tukey_frames: list[pd.DataFrame] = []

    for factor in available_group_cols:
        for value_col in args.value_cols:
            shapiro_df = shapiro_results_for_factor(df, factor, value_col, args.alpha)
            shapiro_frames.append(shapiro_df)

            anova_row, tukey_df = anova_and_tukey_for_factor(df, factor, value_col, args.alpha)
            anova_rows.append(anova_row)
            if not tukey_df.empty:
                tukey_frames.append(tukey_df)

    shapiro_results = pd.concat(shapiro_frames, ignore_index=True)
    normality_summary = summarize_normality(shapiro_results)
    anova_results = pd.DataFrame(anova_rows).sort_values(["factor", "p_value", "outcome"], na_position="last")
    significant_anova = anova_results[anova_results["significant"] == True].copy()
    tukey_results = pd.concat(tukey_frames, ignore_index=True) if tukey_frames else pd.DataFrame()

    output_stem = args.sheet
    workbook_path = output_dir / f"{output_stem}_parametric_analysis.xlsx"
    shapiro_csv = output_dir / f"{output_stem}_shapiro_results.csv"
    anova_csv = output_dir / f"{output_stem}_anova_results.csv"
    tukey_csv = output_dir / f"{output_stem}_tukey_results.csv"

    metadata = pd.DataFrame(
        {
            "item": [
                "input_file",
                "sheet",
                "alpha",
                "group_columns",
                "value_columns",
                "rows_analyzed",
                "skipped_group_columns",
            ],
            "value": [
                str(args.input),
                args.sheet,
                args.alpha,
                ", ".join(available_group_cols),
                ", ".join(args.value_cols),
                len(df),
                ", ".join(skipped_group_cols) if skipped_group_cols else "None",
            ],
        }
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="metadata", index=False)
        normality_summary.to_excel(writer, sheet_name="normality_summary", index=False)
        shapiro_results.to_excel(writer, sheet_name="shapiro", index=False)
        anova_results.to_excel(writer, sheet_name="anova", index=False)
        significant_anova.to_excel(writer, sheet_name="anova_significant", index=False)
        if tukey_results.empty:
            pd.DataFrame(columns=["factor", "outcome"]).to_excel(writer, sheet_name="tukey", index=False)
        else:
            tukey_results.to_excel(writer, sheet_name="tukey", index=False)

    shapiro_results.to_csv(shapiro_csv, index=False)
    anova_results.to_csv(anova_csv, index=False)
    if tukey_results.empty:
        pd.DataFrame(columns=["factor", "outcome"]).to_csv(tukey_csv, index=False)
    else:
        tukey_results.to_csv(tukey_csv, index=False)

    print_console_summary(
        df=df,
        input_file=args.input,
        sheet_name=args.sheet,
        group_cols=available_group_cols,
        value_cols=args.value_cols,
        shapiro_df=shapiro_results,
        anova_df=anova_results,
        tukey_df=tukey_results,
    )
    print("\nSaved outputs:")
    print(f"- {workbook_path}")
    print(f"- {shapiro_csv}")
    print(f"- {anova_csv}")
    print(f"- {tukey_csv}")


if __name__ == "__main__":
    main()
