from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "master_error.xlsx"
DEFAULT_ALPHA = 0.05
DEFAULT_EXCEL_OUTPUT = SCRIPT_DIR / "parametric_assumption_detailed_results.xlsx"


@dataclass(frozen=True)
class AnalysisTest:
    sheet_name: str
    factor: str
    target: str
    name: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or f"{self.sheet_name} | factor={self.factor} | target={self.target}"


@dataclass
class DetailedTestResult:
    test: AnalysisTest
    text_report: str
    summary_row: dict[str, object]
    assumptions_df: pd.DataFrame
    group_diagnostics_df: pd.DataFrame
    analyzed_rows_df: pd.DataFrame


# Edit this list if you want permanent built-in tests.
DEFAULT_TESTS: list[AnalysisTest] = [
    AnalysisTest(sheet_name="adjusted_position", factor="side", target="total_error"),
    AnalysisTest(sheet_name="adjusted_position", factor="arc", target="total_error"),
    AnalysisTest(sheet_name="adjusted_position", factor="collar", target="total_error"),
    AnalysisTest(sheet_name="adjusted_position", factor="target_id", target="total_error"),
    AnalysisTest(sheet_name="adjusted_position", factor="operator", target="total_error"),

    AnalysisTest(sheet_name="adjusted_position", factor="side", target="error_x"),
    AnalysisTest(sheet_name="adjusted_position", factor="arc", target="error_x"),
    AnalysisTest(sheet_name="adjusted_position", factor="collar", target="error_x"),
    AnalysisTest(sheet_name="adjusted_position", factor="target_id", target="error_x"),
    AnalysisTest(sheet_name="adjusted_position", factor="operator", target="error_x"),

    AnalysisTest(sheet_name="adjusted_position", factor="side", target="error_y"),
    AnalysisTest(sheet_name="adjusted_position", factor="arc", target="error_y"),
    AnalysisTest(sheet_name="adjusted_position", factor="collar", target="error_y"),
    AnalysisTest(sheet_name="adjusted_position", factor="target_id", target="error_y"),
    AnalysisTest(sheet_name="adjusted_position", factor="operator", target="error_y"),

    AnalysisTest(sheet_name="adjusted_position", factor="side", target="diff_distance"),
    AnalysisTest(sheet_name="adjusted_position", factor="arc", target="diff_distance"),
    AnalysisTest(sheet_name="adjusted_position", factor="collar", target="diff_distance"),
    AnalysisTest(sheet_name="adjusted_position", factor="target_id", target="diff_distance"),
    AnalysisTest(sheet_name="adjusted_position", factor="operator", target="diff_distance"),

    # Adapter based workflow
    AnalysisTest(sheet_name="adapter_fig", factor="side", target="total_error"),
    AnalysisTest(sheet_name="adapter_fig", factor="arc", target="total_error"),
    AnalysisTest(sheet_name="adapter_fig", factor="collar", target="total_error"),
    AnalysisTest(sheet_name="adapter_fig", factor="target_id", target="total_error"),
    AnalysisTest(sheet_name="adapter_fig", factor="operator", target="total_error"),

    AnalysisTest(sheet_name="adapter_fig", factor="side", target="error_x"),
    AnalysisTest(sheet_name="adapter_fig", factor="arc", target="error_x"),
    AnalysisTest(sheet_name="adapter_fig", factor="collar", target="error_x"),
    AnalysisTest(sheet_name="adapter_fig", factor="target_id", target="error_x"),
    AnalysisTest(sheet_name="adapter_fig", factor="operator", target="error_x"),

    AnalysisTest(sheet_name="adapter_fig", factor="side", target="error_y"),
    AnalysisTest(sheet_name="adapter_fig", factor="arc", target="error_y"),
    AnalysisTest(sheet_name="adapter_fig", factor="collar", target="error_y"),
    AnalysisTest(sheet_name="adapter_fig", factor="target_id", target="error_y"),
    AnalysisTest(sheet_name="adapter_fig", factor="operator", target="error_y"),

    AnalysisTest(sheet_name="adapter_fig", factor="side", target="error_z"),
    AnalysisTest(sheet_name="adapter_fig", factor="arc", target="error_z"),
    AnalysisTest(sheet_name="adapter_fig", factor="collar", target="error_z"),
    AnalysisTest(sheet_name="adapter_fig", factor="target_id", target="error_z"),
    AnalysisTest(sheet_name="adapter_fig", factor="operator", target="error_z"),
    
    # Adapter vs Calculated
    AnalysisTest(sheet_name="comparison", factor="method", target="total_error"),
    AnalysisTest(sheet_name="comparison", factor="method", target="error_x"),
    AnalysisTest(sheet_name="comparison", factor="method", target="error_y"),
    AnalysisTest(sheet_name="comparison", factor="method", target="error_z"),

]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check common parametric-analysis assumptions for one-way group comparisons "
            "stored in master_error.xlsx."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the Excel workbook. Default: figures/master_error.xlsx",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Significance level for Shapiro and Levene tests.",
    )
    parser.add_argument(
        "--test",
        nargs=3,
        action="append",
        metavar=("SHEET_NAME", "FACTOR", "TARGET"),
        help=(
            "Add a test from the command line. Repeat as needed, for example: "
            "--test adjusted_position operator total_error"
        ),
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default="parametric_assumption_report.txt",
        help="Optional text file to save the printed report.",
    )
    parser.add_argument(
        "--excel-output",
        type=Path,
        default=DEFAULT_EXCEL_OUTPUT,
        help="Excel file for detailed intermediate results.",
    )
    parser.add_argument(
        "--describe-input",
        action="store_true",
        help="Print sheet names and column names, then continue with the analysis.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing {label} columns: {missing}")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def load_sheet(input_file: Path, sheet_name: str) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Workbook not found: {input_file}")

    df = pd.read_excel(input_file, sheet_name=sheet_name)
    return clean_columns(df)


def build_tests(args: argparse.Namespace) -> list[AnalysisTest]:
    if not args.test:
        return DEFAULT_TESTS

    return [
        AnalysisTest(sheet_name=sheet_name, factor=factor, target=target)
        for sheet_name, factor, target in args.test
    ]


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def label_from_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def describe_workbook(input_file: Path) -> str:
    workbook = pd.ExcelFile(input_file)
    lines = ["INPUT WORKBOOK", f"- file: {input_file}"]
    for sheet_name in workbook.sheet_names:
        df = clean_columns(pd.read_excel(input_file, sheet_name=sheet_name, nrows=5))
        lines.append(f"- sheet: {sheet_name}")
        lines.append(f"  columns: {', '.join(df.columns)}")
    return "\n".join(lines)


def assess_continuous_dv(values: pd.Series, original_non_null_count: int) -> tuple[str, str, dict[str, object]]:
    non_null = values.dropna()
    if non_null.empty:
        return "NO", "The target column has no usable numeric values.", {
            "numeric_non_null_n": 0,
            "unique_numeric_values": 0,
            "coerced_non_numeric_count": int(original_non_null_count),
        }

    unique_count = int(non_null.nunique(dropna=True))
    summary = {
        "numeric_non_null_n": int(len(non_null)),
        "unique_numeric_values": unique_count,
        "coerced_non_numeric_count": int(original_non_null_count - len(non_null)),
    }
    if len(non_null) < original_non_null_count:
        return (
            "REVIEW",
            "The target column is only partially numeric after coercion, so the DV format needs inspection.",
            summary,
        )

    if unique_count <= min(5, len(non_null)):
        return (
            "REVIEW",
            f"The target column is numeric, but it has only {unique_count} unique values, so it may be discrete rather than continuous.",
            summary,
        )

    return (
        "LIKELY YES",
        f"The target column is numeric with {len(non_null)} usable observations and {unique_count} unique values, which is compatible with a continuous DV.",
        summary,
    )


def assess_independence(df: pd.DataFrame) -> tuple[str, str, dict[str, object]]:
    hint_columns = [column for column in ["target_id", "operator", "trial", "method"] if column in df.columns]
    if hint_columns:
        return (
            "MANUAL CHECK",
            "Independence is a study-design assumption and cannot be verified automatically here. "
            f"Relevant identifier columns present: {', '.join(hint_columns)}.",
            {"independence_hint_columns": ", ".join(hint_columns)},
        )

    return (
        "MANUAL CHECK",
        "Independence is a study-design assumption and cannot be verified automatically from the sheet alone.",
        {"independence_hint_columns": ""},
    )


def shapiro_result(values: np.ndarray, alpha: float) -> dict[str, object]:
    result: dict[str, object] = {
        "n": int(values.size),
        "w_stat": np.nan,
        "p_value": np.nan,
        "ran": False,
        "normal": np.nan,
        "note": "",
    }
    if values.size < 3:
        result["note"] = "Shapiro-Wilk needs at least 3 observations."
        return result

    w_stat, p_value = stats.shapiro(values)
    result.update(
        {
            "w_stat": float(w_stat),
            "p_value": float(p_value),
            "ran": True,
            "normal": bool(p_value >= alpha),
        }
    )
    return result


def tukey_outlier_counts(values: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "iqr_ran": False,
        "mild_outliers": np.nan,
        "extreme_outliers": np.nan,
        "outlier_note": "",
    }
    if values.size < 4:
        result["outlier_note"] = "IQR outlier screening is more stable with at least 4 observations."
        return result

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    inner_low = q1 - 1.5 * iqr
    inner_high = q3 + 1.5 * iqr
    outer_low = q1 - 3.0 * iqr
    outer_high = q3 + 3.0 * iqr

    mild_mask = (values < inner_low) | (values > inner_high)
    extreme_mask = (values < outer_low) | (values > outer_high)
    result.update(
        {
            "iqr_ran": True,
            "mild_outliers": int(mild_mask.sum()),
            "extreme_outliers": int(extreme_mask.sum()),
        }
    )
    return result


def build_group_diagnostics(df: pd.DataFrame, factor: str, target: str, alpha: float) -> pd.DataFrame:
    subset = df[[factor, target]].copy()
    subset[target] = to_numeric(subset[target])
    subset = subset.dropna(subset=[factor, target])

    records: list[dict[str, object]] = []
    for level, group_df in subset.groupby(factor, dropna=True, sort=True):
        values = group_df[target].to_numpy(dtype=float)
        shapiro = shapiro_result(values, alpha)
        outliers = tukey_outlier_counts(values)
        records.append(
            {
                "group": label_from_value(level),
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "shapiro_w": shapiro["w_stat"],
                "shapiro_p": shapiro["p_value"],
                "shapiro_ran": shapiro["ran"],
                "group_normal": shapiro["normal"],
                "group_note": shapiro["note"],
                "iqr_ran": outliers["iqr_ran"],
                "mild_outliers": outliers["mild_outliers"],
                "extreme_outliers": outliers["extreme_outliers"],
                "outlier_note": outliers["outlier_note"],
            }
        )

    return pd.DataFrame.from_records(records)


def assess_normality(subset: pd.DataFrame, factor: str, target: str, group_df: pd.DataFrame, alpha: float) -> tuple[str, str, dict[str, object]]:
    residuals = subset[target] - subset.groupby(factor)[target].transform("mean")
    residual_test = shapiro_result(residuals.dropna().to_numpy(dtype=float), alpha)

    tested_groups = group_df[group_df["shapiro_ran"] == True].copy()
    insufficient_groups = int((group_df["shapiro_ran"] != True).sum()) if not group_df.empty else 0
    non_normal_groups = int((tested_groups["group_normal"] == False).sum()) if not tested_groups.empty else 0

    if residual_test["ran"] and non_normal_groups == 0:
        verdict = "YES" if residual_test["normal"] else "NO"
    elif non_normal_groups > 0:
        verdict = "NO"
    else:
        verdict = "REVIEW"

    residual_text = (
        f"Residual Shapiro-Wilk p={residual_test['p_value']:.4f}" if residual_test["ran"] else residual_test["note"]
    )
    detail = (
        f"{residual_text}. Groups tested={len(tested_groups)}/{len(group_df)}; "
        f"non-normal groups={non_normal_groups}; insufficient-n groups={insufficient_groups}."
    )
    summary = {
        "residual_shapiro_p": residual_test["p_value"],
        "residual_shapiro_ran": residual_test["ran"],
        "residual_normal": residual_test["normal"],
        "groups_tested": int(len(tested_groups)),
        "groups_total": int(len(group_df)),
        "groups_non_normal": non_normal_groups,
        "groups_insufficient_n": insufficient_groups,
    }
    return verdict, detail, summary


def assess_variance_homogeneity(group_df: pd.DataFrame, subset: pd.DataFrame, factor: str, target: str, alpha: float) -> tuple[str, str, dict[str, object]]:
    eligible_levels = group_df.loc[group_df["n"] >= 2, "group"].tolist() if not group_df.empty else []
    arrays: list[np.ndarray] = []
    for level in eligible_levels:
        values = subset.loc[subset["_group_label"] == level, target].to_numpy(dtype=float)
        if values.size >= 2:
            arrays.append(values)

    if len(arrays) < 2:
        return (
            "REVIEW",
            "Levene/Brown-Forsythe needs at least 2 groups with at least 2 observations each.",
            {"levene_stat": np.nan, "levene_p": np.nan, "groups_used": len(arrays)},
        )

    stat, p_value = stats.levene(*arrays, center="median")
    verdict = "YES" if p_value >= alpha else "NO"
    detail = (
        f"Brown-Forsythe/Levene p={p_value:.4f} using {len(arrays)} groups with n>=2. "
        + ("Variances look similar." if verdict == "YES" else "Heteroscedasticity is likely.")
    )
    return verdict, detail, {"levene_stat": float(stat), "levene_p": float(p_value), "groups_used": len(arrays)}


def assess_outliers(group_df: pd.DataFrame) -> tuple[str, str, dict[str, object]]:
    if group_df.empty:
        return "REVIEW", "No analyzable groups were available.", {"groups_flagged": 0, "extreme_total": 0}

    evaluable = group_df[group_df["iqr_ran"] == True].copy()
    if evaluable.empty:
        return (
            "REVIEW",
            "No groups had enough observations for stable IQR outlier screening.",
            {"groups_flagged": 0, "extreme_total": 0},
        )

    extreme_total = int(evaluable["extreme_outliers"].fillna(0).sum())
    flagged_groups = evaluable.loc[evaluable["extreme_outliers"].fillna(0) > 0, "group"].tolist()

    if extreme_total == 0:
        return (
            "YES",
            f"No extreme Tukey outliers were detected in {len(evaluable)} evaluable groups.",
            {"groups_flagged": 0, "extreme_total": 0},
        )

    return (
        "REVIEW",
        f"Extreme Tukey outliers were detected: total={extreme_total}; affected groups={', '.join(flagged_groups)}.",
        {"groups_flagged": len(flagged_groups), "extreme_total": extreme_total},
    )


def suggest_next_step(normality_verdict: str, variance_verdict: str, outlier_verdict: str) -> str:
    if normality_verdict == "YES" and variance_verdict == "YES" and outlier_verdict == "YES":
        return "A standard one-way ANOVA is defensible if the design is also independent."
    if normality_verdict == "YES" and variance_verdict == "NO":
        return "Normality looks acceptable, but unequal variances suggest Welch ANOVA instead of standard ANOVA."
    return "Parametric assumptions are not fully supported. Consider transformation, a robust method, or Kruskal-Wallis depending on the design."


def format_assumption_lines(answers: list[tuple[str, str, str]]) -> list[str]:
    lines = ["ASSUMPTION CHECKS"]
    for question, verdict, detail in answers:
        lines.append(f"- {question}: {verdict}")
        lines.append(f"  {detail}")
    return lines


def format_group_table(group_df: pd.DataFrame) -> list[str]:
    if group_df.empty:
        return ["GROUP DIAGNOSTICS", "- No analyzable groups."]

    display_df = group_df.copy()
    numeric_cols = [
        "mean",
        "median",
        "std",
        "min",
        "max",
        "shapiro_w",
        "shapiro_p",
    ]
    for column in numeric_cols:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    for column in ["mild_outliers", "extreme_outliers"]:
        display_df[column] = display_df[column].map(lambda value: "" if pd.isna(value) else str(int(value)))

    display_df = display_df[
        [
            "group",
            "n",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "shapiro_p",
            "mild_outliers",
            "extreme_outliers",
            "group_note",
            "outlier_note",
        ]
    ]

    return ["GROUP DIAGNOSTICS", display_df.to_string(index=False)]


def build_assumptions_dataframe(test: AnalysisTest, answers: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for question, verdict, detail in answers:
        rows.append(
            {
                "test_name": test.display_name,
                "sheet_name": test.sheet_name,
                "factor": test.factor,
                "target": test.target,
                "assumption": question,
                "verdict": verdict,
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def prepare_analyzed_rows(test: AnalysisTest, subset: pd.DataFrame) -> pd.DataFrame:
    analyzed_rows_df = subset[[test.factor, "_group_label", test.target]].copy()
    analyzed_rows_df = analyzed_rows_df.rename(
        columns={
            test.factor: "factor_value",
            "_group_label": "group_label",
            test.target: "target_value",
        }
    )
    analyzed_rows_df.insert(0, "target", test.target)
    analyzed_rows_df.insert(0, "factor", test.factor)
    analyzed_rows_df.insert(0, "sheet_name", test.sheet_name)
    analyzed_rows_df.insert(0, "test_name", test.display_name)
    return analyzed_rows_df.reset_index(drop=True)


def save_detailed_results(
    output_file: Path,
    results: list[DetailedTestResult],
    error_records: list[dict[str, object]],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([result.summary_row for result in results])
    assumptions_df = pd.concat([result.assumptions_df for result in results], ignore_index=True) if results else pd.DataFrame()
    group_diagnostics_df = pd.concat([result.group_diagnostics_df for result in results], ignore_index=True) if results else pd.DataFrame()
    analyzed_rows_df = pd.concat([result.analyzed_rows_df for result in results], ignore_index=True) if results else pd.DataFrame()
    errors_df = pd.DataFrame(error_records)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        assumptions_df.to_excel(writer, sheet_name="assumptions", index=False)
        group_diagnostics_df.to_excel(writer, sheet_name="group_diagnostics", index=False)
        analyzed_rows_df.to_excel(writer, sheet_name="analyzed_rows", index=False)
        errors_df.to_excel(writer, sheet_name="errors", index=False)


def analyze_test(df: pd.DataFrame, test: AnalysisTest, alpha: float) -> DetailedTestResult:
    require_columns(df, [test.factor, test.target], "analysis")

    rows_before_drop = int(len(df))
    subset = df[[test.factor, test.target]].copy()
    original_non_null_count = int(subset[test.target].notna().sum())
    subset[test.target] = to_numeric(subset[test.target])
    subset = subset.dropna(subset=[test.factor, test.target]).copy()
    subset["_group_label"] = subset[test.factor].map(label_from_value)

    if subset.empty:
        raise ValueError(
            f"No analyzable rows remain for sheet='{test.sheet_name}', factor='{test.factor}', target='{test.target}'."
        )

    group_df = build_group_diagnostics(subset, factor=test.factor, target=test.target, alpha=alpha)

    continuous_verdict, continuous_detail, continuous_summary = assess_continuous_dv(
        values=to_numeric(df[test.target]),
        original_non_null_count=original_non_null_count,
    )
    independence_verdict, independence_detail, independence_summary = assess_independence(df)
    normality_verdict, normality_detail, normality_summary = assess_normality(subset, test.factor, test.target, group_df, alpha)
    variance_verdict, variance_detail, variance_summary = assess_variance_homogeneity(group_df, subset, test.factor, test.target, alpha)
    outlier_verdict, outlier_detail, outlier_summary = assess_outliers(group_df)
    next_step = suggest_next_step(normality_verdict, variance_verdict, outlier_verdict)

    answers = [
        ("DV is continuous (interval/ratio)", continuous_verdict, continuous_detail),
        ("Observations are independent", independence_verdict, independence_detail),
        ("Within-group distributions are roughly normal (or residuals are)", normality_verdict, normality_detail),
        ("Group variances are similar", variance_verdict, variance_detail),
        ("No extreme outliers dominate the groups", outlier_verdict, outlier_detail),
    ]

    lines = [
        "=" * 100,
        f"TEST: {test.display_name}",
        f"Rows analyzed: {len(subset)}",
        f"Groups analyzed: {subset['_group_label'].nunique()}",
        f"Alpha: {alpha}",
        *format_assumption_lines(answers),
        "NEXT STEP",
        f"- {next_step}",
        *format_group_table(group_df),
    ]

    assumptions_df = build_assumptions_dataframe(test, answers)
    group_diagnostics_df = group_df.copy()
    if not group_diagnostics_df.empty:
        group_diagnostics_df.insert(0, "target", test.target)
        group_diagnostics_df.insert(0, "factor", test.factor)
        group_diagnostics_df.insert(0, "sheet_name", test.sheet_name)
        group_diagnostics_df.insert(0, "test_name", test.display_name)

    summary_row = {
        "test_name": test.display_name,
        "sheet_name": test.sheet_name,
        "factor": test.factor,
        "target": test.target,
        "alpha": alpha,
        "rows_in_sheet": rows_before_drop,
        "rows_analyzed": int(len(subset)),
        "rows_dropped": int(rows_before_drop - len(subset)),
        "groups_analyzed": int(subset["_group_label"].nunique()),
        "dv_continuous_verdict": continuous_verdict,
        "dv_continuous_detail": continuous_detail,
        "independence_verdict": independence_verdict,
        "independence_detail": independence_detail,
        "normality_verdict": normality_verdict,
        "normality_detail": normality_detail,
        "variance_verdict": variance_verdict,
        "variance_detail": variance_detail,
        "outlier_verdict": outlier_verdict,
        "outlier_detail": outlier_detail,
        "recommended_next_step": next_step,
        **continuous_summary,
        **independence_summary,
        **normality_summary,
        **variance_summary,
        **outlier_summary,
    }

    return DetailedTestResult(
        test=test,
        text_report="\n".join(lines),
        summary_row=summary_row,
        assumptions_df=assumptions_df,
        group_diagnostics_df=group_diagnostics_df,
        analyzed_rows_df=prepare_analyzed_rows(test, subset),
    )


def main() -> None:
    args = parse_args()
    tests = build_tests(args)

    report_sections: list[str] = []
    detailed_results: list[DetailedTestResult] = []
    error_records: list[dict[str, object]] = []
    if args.describe_input:
        report_sections.append(describe_workbook(args.input))

    workbook = pd.ExcelFile(args.input)
    sheet_cache: dict[str, pd.DataFrame] = {}

    for test in tests:
        if test.sheet_name not in workbook.sheet_names:
            message = f"Sheet '{test.sheet_name}' does not exist. Available sheets: {workbook.sheet_names}"
            error_records.append(
                {
                    "test_name": test.display_name,
                    "sheet_name": test.sheet_name,
                    "factor": test.factor,
                    "target": test.target,
                    "error": message,
                }
            )
            report_sections.append("\n".join(["=" * 100, f"TEST: {test.display_name}", f"ERROR: {message}"]))
            continue

        try:
            if test.sheet_name not in sheet_cache:
                sheet_cache[test.sheet_name] = load_sheet(args.input, test.sheet_name)
            result = analyze_test(sheet_cache[test.sheet_name], test=test, alpha=args.alpha)
            detailed_results.append(result)
            report_sections.append(result.text_report)
        except Exception as exc:
            error_records.append(
                {
                    "test_name": test.display_name,
                    "sheet_name": test.sheet_name,
                    "factor": test.factor,
                    "target": test.target,
                    "error": str(exc),
                }
            )
            report_sections.append("\n".join(["=" * 100, f"TEST: {test.display_name}", f"ERROR: {exc}"]))

    report = "\n\n".join(report_sections)
    print(report)

    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(report, encoding="utf-8")
        print(f"\nSaved report to: {args.report_file}")

    if args.excel_output is not None:
        save_detailed_results(args.excel_output, detailed_results, error_records)
        print(f"Saved detailed Excel results to: {args.excel_output}")


if __name__ == "__main__":
    main()
