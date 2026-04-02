from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "master_error.xlsx"
DEFAULT_ALPHA = 0.05
DEFAULT_TEXT_OUTPUT = SCRIPT_DIR / "configurable_statistical_tests_report.txt"
DEFAULT_EXCEL_OUTPUT = SCRIPT_DIR / "configurable_statistical_tests_results.xlsx"

SUPPORTED_ANALYSES = {
    "paired t-test": "Paired t-test",
    "paired t test": "Paired t-test",
    "wilcoxon signed-rank": "Wilcoxon signed-rank",
    "wilcoxon signed rank": "Wilcoxon signed-rank",
    "repeated-measures anova": "Repeated-measures ANOVA",
    "repeated measures anova": "Repeated-measures ANOVA",
    "friedman test": "Friedman test",
    "two-sample t-test (pooled)": "Two-sample t-test (pooled)",
    "two sample t-test (pooled)": "Two-sample t-test (pooled)",
    "two-sample t test (pooled)": "Two-sample t-test (pooled)",
    "welch's t-test": "Welch's t-test",
    "welchs t-test": "Welch's t-test",
    "welch t-test": "Welch's t-test",
    "mann-whitney (wilcoxon rank-sum)": "Mann-Whitney (Wilcoxon rank-sum)",
    "mann whitney (wilcoxon rank-sum)": "Mann-Whitney (Wilcoxon rank-sum)",
    "mann-whitney": "Mann-Whitney (Wilcoxon rank-sum)",
    "one-way anova": "One-way ANOVA",
    "one way anova": "One-way ANOVA",
    "welch anova": "Welch ANOVA",
    "kruskal-wallis": "Kruskal-Wallis",
    "kruskal wallis": "Kruskal-Wallis",
}

SUBJECT_BASED_ANALYSES = {
    "Paired t-test",
    "Wilcoxon signed-rank",
    "Repeated-measures ANOVA",
    "Friedman test",
}

TWO_LEVEL_ANALYSES = {
    "Paired t-test",
    "Wilcoxon signed-rank",
    "Two-sample t-test (pooled)",
    "Welch's t-test",
    "Mann-Whitney (Wilcoxon rank-sum)",
}

OMNIBUS_ANALYSES = {
    "One-way ANOVA",
    "Welch ANOVA",
    "Repeated-measures ANOVA",
    "Kruskal-Wallis",
    "Friedman test",
}


@dataclass(frozen=True)
class AnalysisTest:
    analysis_name: str
    sheet_name: str
    factor: str
    target: str
    subject: str | None = None
    levels: tuple[str, ...] | None = None
    name: str | None = None

    @property
    def canonical_analysis_name(self) -> str:
        return canonicalize_analysis_name(self.analysis_name)

    @property
    def display_name(self) -> str:
        return self.name or (
            f"{self.canonical_analysis_name} | {self.sheet_name} | "
            f"factor={self.factor} | target={self.target}"
        )


@dataclass
class DetailedTestResult:
    test: AnalysisTest
    text_report: str
    summary_row: dict[str, object]
    assumptions_df: pd.DataFrame
    group_diagnostics_df: pd.DataFrame
    analyzed_rows_df: pd.DataFrame
    analysis_results_df: pd.DataFrame
    analysis_input_df: pd.DataFrame
    posthoc_results_df: pd.DataFrame


# Edit this list to define the exact tests to run.
#
# For paired / repeated-measures analyses, set `subject`.
# For any 2-level test where the factor has more than 2 levels, set `levels`.
DEFAULT_TESTS: list[AnalysisTest] = [
    AnalysisTest(analysis_name="Two-sample t-test (pooled)", sheet_name="adjusted_position", factor="side", target="total_error", name="Adjusted-position total error by side"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="arc", target="total_error", name="Adjusted-position total error by arc"),
    AnalysisTest(analysis_name="Mann-Whitney (Wilcoxon rank-sum)", sheet_name="adjusted_position", factor="collar", target="total_error", name="Adjusted-position total error by collar"),
    AnalysisTest(analysis_name="Kruskal-Wallis", sheet_name="adjusted_position", factor="target_id", target="total_error", name="Adjusted-position total error by target_id"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="operator", target="total_error", name="Adjusted-position total error by operator"),

    AnalysisTest(analysis_name="Two-sample t-test (pooled)", sheet_name="adjusted_position", factor="side", target="error_x", name="Adjusted-position error_x by side"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="arc", target="error_x", name="Adjusted-position error_x by arc"),
    AnalysisTest(analysis_name="Two-sample t-test (pooled)", sheet_name="adjusted_position", factor="collar", target="error_x", name="Adjusted-position error_x by collar"),
    AnalysisTest(analysis_name="Kruskal-Wallis", sheet_name="adjusted_position", factor="target_id", target="error_x", name="Adjusted-position error_x by target_id"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="operator", target="error_x", name="Adjusted-position error_x by operator"),

    AnalysisTest(analysis_name="Mann-Whitney (Wilcoxon rank-sum)", sheet_name="adjusted_position", factor="side", target="error_y", name="Adjusted-position error_y by side"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="arc", target="error_y", name="Adjusted-position error_y by arc"),
    AnalysisTest(analysis_name="Mann-Whitney (Wilcoxon rank-sum)", sheet_name="adjusted_position", factor="collar", target="error_y", name="Adjusted-position error_y by collar"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="target_id", target="error_y", name="Adjusted-position error_y by target_id"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="operator", target="error_y", name="Adjusted-position error_y by operator"),

    AnalysisTest(analysis_name="Two-sample t-test (pooled)", sheet_name="adjusted_position", factor="side", target="diff_distance", name="Adjusted-position diff_distance by side"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="arc", target="diff_distance", name="Adjusted-position diff_distance by arc"),
    AnalysisTest(analysis_name="Two-sample t-test (pooled)", sheet_name="adjusted_position", factor="collar", target="diff_distance", name="Adjusted-position diff_distance by collar"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="target_id", target="diff_distance", name="Adjusted-position diff_distance by target_id"),
    AnalysisTest(analysis_name="One-way ANOVA", sheet_name="adjusted_position", factor="operator", target="diff_distance", name="Adjusted-position diff_distance by operator"),

    AnalysisTest(
        analysis_name="Paired t-test",
        sheet_name="comparison",
        factor="method",
        target="total_error",
        subject="target_id",
        levels=("adapter", "adjusted"),
        name="Comparison sheet paired total error by method",
    ),
    AnalysisTest(
        analysis_name="Paired t-test",
        sheet_name="comparison",
        factor="method",
        target="error_x",
        subject="target_id",
        levels=("adapter", "adjusted"),
        name="Comparison sheet paired error_x by method",
    ),
    AnalysisTest(
        analysis_name="Wilcoxon signed-rank",
        sheet_name="comparison",
        factor="method",
        target="error_y",
        subject="target_id",
        levels=("adapter", "adjusted"),
        name="Comparison sheet paired error_y by method",
    ),
    AnalysisTest(
        analysis_name="Paired t-test",
        sheet_name="comparison",
        factor="method",
        target="error_z",
        subject="target_id",
        levels=("adapter", "adjusted"),
        name="Comparison sheet paired error_z by method",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run configurable statistical tests plus assumption checks on master_error.xlsx. "
            "Edit DEFAULT_TESTS in this file to choose analyses."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the Excel workbook.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Significance level for the analysis and assumption checks.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=DEFAULT_TEXT_OUTPUT,
        help="Text file for the summary report.",
    )
    parser.add_argument(
        "--excel-output",
        type=Path,
        default=DEFAULT_EXCEL_OUTPUT,
        help="Excel file for detailed results.",
    )
    parser.add_argument(
        "--describe-input",
        action="store_true",
        help="Print workbook sheet names and columns before analysis.",
    )
    return parser.parse_args()


def canonicalize_analysis_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("’", "'")
    if normalized not in SUPPORTED_ANALYSES:
        supported = ", ".join(sorted(set(SUPPORTED_ANALYSES.values())))
        raise ValueError(f"Unsupported analysis '{name}'. Supported values: {supported}")
    return SUPPORTED_ANALYSES[normalized]


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
    return clean_columns(pd.read_excel(input_file, sheet_name=sheet_name))


def build_tests() -> list[AnalysisTest]:
    return DEFAULT_TESTS


def describe_workbook(input_file: Path) -> str:
    workbook = pd.ExcelFile(input_file)
    lines = ["INPUT WORKBOOK", f"- file: {input_file}"]
    for sheet_name in workbook.sheet_names:
        df = clean_columns(pd.read_excel(input_file, sheet_name=sheet_name, nrows=5))
        lines.append(f"- sheet: {sheet_name}")
        lines.append(f"  columns: {', '.join(df.columns)}")
    return "\n".join(lines)


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


def format_float(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


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


def assess_independence(df: pd.DataFrame, subject: str | None) -> tuple[str, str, dict[str, object]]:
    hint_columns = [column for column in ["target_id", "operator", "trial", "method", subject] if column and column in df.columns]
    hint_columns = list(dict.fromkeys(hint_columns))
    if hint_columns:
        return (
            "MANUAL CHECK",
            "Independence is a design assumption and cannot be verified automatically here. "
            f"Relevant identifier columns present: {', '.join(hint_columns)}.",
            {"independence_hint_columns": ", ".join(hint_columns)},
        )

    return (
        "MANUAL CHECK",
        "Independence is a design assumption and cannot be verified automatically from the sheet alone.",
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


def build_group_diagnostics(df: pd.DataFrame, factor_label_col: str, target: str, alpha: float) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for level, group_df in df.groupby(factor_label_col, dropna=True, sort=True):
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


def assess_normality(subset: pd.DataFrame, factor_label_col: str, target: str, group_df: pd.DataFrame, alpha: float) -> tuple[str, str, dict[str, object]]:
    residuals = subset[target] - subset.groupby(factor_label_col)[target].transform("mean")
    residual_test = shapiro_result(residuals.dropna().to_numpy(dtype=float), alpha)

    tested_groups = group_df[group_df["shapiro_ran"] == True].copy() if not group_df.empty else pd.DataFrame()
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
        "residual_shapiro_w": residual_test["w_stat"],
        "residual_shapiro_p": residual_test["p_value"],
        "residual_shapiro_ran": residual_test["ran"],
        "residual_normal": residual_test["normal"],
        "groups_tested": int(len(tested_groups)),
        "groups_total": int(len(group_df)),
        "groups_non_normal": non_normal_groups,
        "groups_insufficient_n": insufficient_groups,
    }
    return verdict, detail, summary


def assess_variance_homogeneity(group_df: pd.DataFrame, subset: pd.DataFrame, factor_label_col: str, target: str, alpha: float) -> tuple[str, str, dict[str, object]]:
    eligible_levels = group_df.loc[group_df["n"] >= 2, "group"].tolist() if not group_df.empty else []
    arrays: list[np.ndarray] = []
    for level in eligible_levels:
        values = subset.loc[subset[factor_label_col] == level, target].to_numpy(dtype=float)
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


def suggest_next_step(analysis_name: str, normality_verdict: str, variance_verdict: str, outlier_verdict: str) -> str:
    if analysis_name in {"One-way ANOVA", "Two-sample t-test (pooled)", "Paired t-test", "Repeated-measures ANOVA"}:
        if normality_verdict == "YES" and variance_verdict == "YES" and outlier_verdict == "YES":
            return f"The requested parametric test ({analysis_name}) is reasonably supported if the design assumption is appropriate."
        if analysis_name in {"One-way ANOVA", "Two-sample t-test (pooled)"} and variance_verdict == "NO":
            return "Variance equality looks weak. Consider Welch's alternative if the design is independent."
        return "Some parametric assumptions look weak. Compare with a robust or non-parametric alternative."
    return f"The requested analysis ({analysis_name}) is non-parametric or variance-robust, but the assumption summary is still useful for interpretation."


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
    numeric_cols = ["mean", "median", "std", "min", "max", "shapiro_w", "shapiro_p"]
    for column in numeric_cols:
        display_df[column] = display_df[column].map(lambda value: format_float(value))
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
                "analysis_name": test.canonical_analysis_name,
                "sheet_name": test.sheet_name,
                "factor": test.factor,
                "target": test.target,
                "subject": test.subject,
                "assumption": question,
                "verdict": verdict,
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def prepare_analyzed_rows(test: AnalysisTest, subset: pd.DataFrame) -> pd.DataFrame:
    columns = [test.factor, "_group_label", test.target]
    if test.subject is not None and test.subject in subset.columns:
        columns.insert(0, test.subject)
    analyzed_rows_df = subset[columns].copy()
    rename_map = {
        test.factor: "factor_value",
        "_group_label": "group_label",
        test.target: "target_value",
    }
    if test.subject is not None and test.subject in analyzed_rows_df.columns:
        rename_map[test.subject] = "subject_value"
    analyzed_rows_df = analyzed_rows_df.rename(columns=rename_map)
    analyzed_rows_df.insert(0, "target", test.target)
    analyzed_rows_df.insert(0, "factor", test.factor)
    analyzed_rows_df.insert(0, "subject", test.subject)
    analyzed_rows_df.insert(0, "sheet_name", test.sheet_name)
    analyzed_rows_df.insert(0, "analysis_name", test.canonical_analysis_name)
    analyzed_rows_df.insert(0, "test_name", test.display_name)
    return analyzed_rows_df.reset_index(drop=True)


def prepare_subset(df: pd.DataFrame, test: AnalysisTest) -> tuple[pd.DataFrame, int, list[str], int]:
    analysis_name = test.canonical_analysis_name
    required_columns = [test.factor, test.target]
    if analysis_name in SUBJECT_BASED_ANALYSES:
        if not test.subject:
            raise ValueError(f"{analysis_name} requires `subject` in the AnalysisTest definition.")
        required_columns.append(test.subject)

    require_columns(df, required_columns, "analysis")

    subset = df[required_columns].copy()
    rows_before_drop = int(len(subset))
    original_non_null_count = int(subset[test.target].notna().sum())
    subset[test.target] = to_numeric(subset[test.target])
    subset = subset.dropna(subset=[test.factor, test.target]).copy()
    if test.subject is not None and test.subject in subset.columns:
        subset = subset.dropna(subset=[test.subject]).copy()
    subset["_group_label"] = subset[test.factor].map(label_from_value)

    if test.levels:
        requested_levels = [label_from_value(level) for level in test.levels]
        subset = subset[subset["_group_label"].isin(requested_levels)].copy()
        present_requested_levels = [level for level in requested_levels if level in set(subset["_group_label"])]
        ordered_levels = present_requested_levels
    else:
        ordered_levels = sorted(subset["_group_label"].dropna().unique().tolist())

    if subset.empty:
        raise ValueError(
            f"No analyzable rows remain for analysis='{analysis_name}', sheet='{test.sheet_name}', factor='{test.factor}', target='{test.target}'."
        )

    subset["_group_label"] = pd.Categorical(subset["_group_label"], categories=ordered_levels, ordered=True)
    subset = subset.sort_values(by=["_group_label"]).copy()

    realized_levels = [str(level) for level in subset["_group_label"].dropna().cat.categories if level in set(subset["_group_label"].astype(str))]
    if not realized_levels:
        realized_levels = sorted(subset["_group_label"].astype(str).unique().tolist())

    validate_test_structure(test, realized_levels)
    return subset, rows_before_drop, realized_levels, original_non_null_count


def validate_test_structure(test: AnalysisTest, realized_levels: list[str]) -> None:
    analysis_name = test.canonical_analysis_name
    if analysis_name in SUBJECT_BASED_ANALYSES and not test.subject:
        raise ValueError(f"{analysis_name} requires a subject column.")
    if analysis_name in TWO_LEVEL_ANALYSES and len(realized_levels) != 2:
        raise ValueError(
            f"{analysis_name} requires exactly 2 factor levels after filtering, but found {len(realized_levels)}: {realized_levels}"
        )
    if analysis_name not in TWO_LEVEL_ANALYSES and len(realized_levels) < 2:
        raise ValueError(f"{analysis_name} requires at least 2 factor levels, but found {len(realized_levels)}.")


def build_independent_groups(subset: pd.DataFrame, target: str) -> tuple[list[str], list[np.ndarray]]:
    labels: list[str] = []
    arrays: list[np.ndarray] = []
    for label, group_df in subset.groupby("_group_label", dropna=True, sort=True):
        values = group_df[target].to_numpy(dtype=float)
        if values.size == 0:
            continue
        labels.append(str(label))
        arrays.append(values)
    return labels, arrays


def build_repeated_matrix(test: AnalysisTest, subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if test.subject is None:
        raise ValueError(f"{test.canonical_analysis_name} requires a subject column.")

    duplicate_counts = subset.groupby([test.subject, "_group_label"]).size().reset_index(name="cell_n")
    duplicate_cells = int((duplicate_counts["cell_n"] > 1).sum())

    aggregated = (
        subset.groupby([test.subject, "_group_label"], observed=True, as_index=False)[test.target]
        .mean()
        .rename(columns={test.target: "analysis_value"})
    )
    aggregated["_group_label"] = aggregated["_group_label"].astype(str)

    wide_df = aggregated.pivot(index=test.subject, columns="_group_label", values="analysis_value")
    if test.levels:
        ordered_columns = [label_from_value(level) for level in test.levels if label_from_value(level) in wide_df.columns]
        wide_df = wide_df.reindex(columns=ordered_columns)
    else:
        wide_df = wide_df.reindex(sorted(wide_df.columns), axis=1)

    subjects_before = int(wide_df.shape[0])
    complete_wide_df = wide_df.dropna().copy()
    subjects_after = int(complete_wide_df.shape[0])

    summary = {
        "subjects_before_complete_case": subjects_before,
        "subjects_used_in_analysis": subjects_after,
        "subjects_dropped_incomplete": int(subjects_before - subjects_after),
        "duplicate_subject_factor_cells": duplicate_cells,
    }
    return aggregated, complete_wide_df, summary


def make_analysis_input_df(test: AnalysisTest, analysis_input: pd.DataFrame, input_kind: str) -> pd.DataFrame:
    df = analysis_input.copy().reset_index(drop=False)
    df.insert(0, "input_kind", input_kind)
    df.insert(0, "target", test.target)
    df.insert(0, "factor", test.factor)
    df.insert(0, "subject", test.subject)
    df.insert(0, "sheet_name", test.sheet_name)
    df.insert(0, "analysis_name", test.canonical_analysis_name)
    df.insert(0, "test_name", test.display_name)
    return df


def analysis_result_row_base(test: AnalysisTest, alpha: float, levels: list[str]) -> dict[str, object]:
    return {
        "test_name": test.display_name,
        "analysis_name": test.canonical_analysis_name,
        "sheet_name": test.sheet_name,
        "factor": test.factor,
        "target": test.target,
        "subject": test.subject,
        "levels": " | ".join(levels),
        "alpha": alpha,
    }


def build_posthoc_result_rows_from_matrix(
    matrix_df: pd.DataFrame,
    test: AnalysisTest,
    method: str,
    adjusted_p_column: str,
    alpha: float,
    value_summary: str,
    notes: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = [str(label) for label in matrix_df.index.tolist()]
    for group_a, group_b in combinations(labels, 2):
        adjusted_p = pd.to_numeric(matrix_df.loc[group_a, group_b], errors="coerce")
        rows.append(
            {
                "test_name": test.display_name,
                "analysis_name": test.canonical_analysis_name,
                "sheet_name": test.sheet_name,
                "factor": test.factor,
                "target": test.target,
                "subject": test.subject,
                "posthoc_method": method,
                "group_a": group_a,
                "group_b": group_b,
                "raw_p": np.nan,
                adjusted_p_column: float(adjusted_p) if pd.notna(adjusted_p) else np.nan,
                "significant": bool(adjusted_p < alpha) if pd.notna(adjusted_p) else False,
                "comparison_summary": value_summary,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def run_posthoc_analysis(
    test: AnalysisTest,
    subset: pd.DataFrame,
    alpha: float,
    levels: list[str],
    analysis_result_row: dict[str, object],
) -> tuple[pd.DataFrame, str]:
    analysis_name = test.canonical_analysis_name
    analysis_p = pd.to_numeric(analysis_result_row.get("p_value"), errors="coerce")

    if analysis_name not in OMNIBUS_ANALYSES:
        return pd.DataFrame(), "No omnibus post-hoc analysis is defined for this requested test."

    if pd.isna(analysis_p):
        return pd.DataFrame(), "Post-hoc analysis skipped because the omnibus p-value is unavailable."

    if analysis_p >= alpha:
        return pd.DataFrame(), f"Post-hoc analysis skipped because omnibus p={format_float(analysis_p)} is not below alpha={alpha}."

    if len(levels) <= 2:
        return pd.DataFrame(), "Post-hoc analysis skipped because only 2 levels are present."

    if analysis_name == "One-way ANOVA":
        tukey = pairwise_tukeyhsd(
            endog=subset[test.target].to_numpy(dtype=float),
            groups=subset["_group_label"].astype(str).to_numpy(),
            alpha=alpha,
        )
        summary_rows = tukey.summary().data
        posthoc_df = pd.DataFrame(summary_rows[1:], columns=summary_rows[0]).rename(columns={"p-adj": "p_adj"})
        for column in ["meandiff", "p_adj", "lower", "upper"]:
            if column in posthoc_df.columns:
                posthoc_df[column] = pd.to_numeric(posthoc_df[column], errors="coerce")
        if "reject" in posthoc_df.columns:
            posthoc_df["significant"] = posthoc_df["reject"].astype(str).str.lower().eq("true")
            posthoc_df = posthoc_df.drop(columns=["reject"])
        posthoc_df.insert(0, "posthoc_method", "Tukey HSD")
        posthoc_df.insert(0, "subject", test.subject)
        posthoc_df.insert(0, "target", test.target)
        posthoc_df.insert(0, "factor", test.factor)
        posthoc_df.insert(0, "sheet_name", test.sheet_name)
        posthoc_df.insert(0, "analysis_name", analysis_name)
        posthoc_df.insert(0, "test_name", test.display_name)
        return posthoc_df, "Post-hoc run: Tukey HSD after significant one-way ANOVA."

    if analysis_name == "Welch ANOVA":
        if not hasattr(sp, "posthoc_tamhane"):
            return pd.DataFrame(), "Post-hoc skipped because the installed scikit-posthocs version does not provide Tamhane-style unequal-variance comparisons."
        matrix_df = sp.posthoc_tamhane(
            subset[["_group_label", test.target]].assign(_group_label=subset["_group_label"].astype(str)),
            val_col=test.target,
            group_col="_group_label",
        )
        posthoc_df = build_posthoc_result_rows_from_matrix(
            matrix_df=matrix_df,
            test=test,
            method="Tamhane T2",
            adjusted_p_column="adjusted_p",
            alpha=alpha,
            value_summary="Unequal-variance pairwise comparison",
            notes="Tamhane T2 used as a Games-Howell / Dunnett-T3-type unequal-variance post-hoc procedure.",
        )
        return posthoc_df, "Post-hoc run: Tamhane T2 after significant Welch ANOVA."

    if analysis_name == "Repeated-measures ANOVA":
        _, wide_df, repeated_summary = build_repeated_matrix(test, subset)
        if wide_df.empty:
            return pd.DataFrame(), "Post-hoc skipped because no complete-case subjects remained."
        pair_rows: list[dict[str, object]] = []
        raw_p_values: list[float] = []
        pairs = list(combinations(list(wide_df.columns), 2))
        for group_a, group_b in pairs:
            statistic, p_value = stats.ttest_rel(
                wide_df[group_a].to_numpy(dtype=float),
                wide_df[group_b].to_numpy(dtype=float),
            )
            raw_p_values.append(float(p_value))
            diff = wide_df[group_a] - wide_df[group_b]
            pair_rows.append(
                {
                    "test_name": test.display_name,
                    "analysis_name": analysis_name,
                    "sheet_name": test.sheet_name,
                    "factor": test.factor,
                    "target": test.target,
                    "subject": test.subject,
                    "posthoc_method": "Paired t-test + Holm",
                    "group_a": str(group_a),
                    "group_b": str(group_b),
                    "statistic": float(statistic),
                    "raw_p": float(p_value),
                    "mean_difference": float(diff.mean()),
                    "n_used": int(len(diff)),
                    **repeated_summary,
                    "notes": "Pairwise paired t-tests with Holm correction.",
                }
            )
        _, adjusted_p_values, _, _ = multipletests(raw_p_values, alpha=alpha, method="holm")
        for row, adjusted_p in zip(pair_rows, adjusted_p_values):
            row["adjusted_p"] = float(adjusted_p)
            row["significant"] = bool(adjusted_p < alpha)
        return pd.DataFrame(pair_rows), "Post-hoc run: pairwise paired t-tests with Holm correction after significant repeated-measures ANOVA."

    if analysis_name == "Kruskal-Wallis":
        matrix_df = sp.posthoc_dunn(
            subset[["_group_label", test.target]].assign(_group_label=subset["_group_label"].astype(str)),
            val_col=test.target,
            group_col="_group_label",
            p_adjust="holm",
        )
        posthoc_df = build_posthoc_result_rows_from_matrix(
            matrix_df=matrix_df,
            test=test,
            method="Dunn + Holm",
            adjusted_p_column="adjusted_p",
            alpha=alpha,
            value_summary="Rank-based pairwise comparison",
            notes="Dunn pairwise comparisons with Holm correction.",
        )
        return posthoc_df, "Post-hoc run: Dunn pairwise comparisons with Holm correction after significant Kruskal-Wallis."

    if analysis_name == "Friedman test":
        _, wide_df, repeated_summary = build_repeated_matrix(test, subset)
        if wide_df.empty:
            return pd.DataFrame(), "Post-hoc skipped because no complete-case subjects remained."
        pair_rows: list[dict[str, object]] = []
        raw_p_values: list[float] = []
        pairs = list(combinations(list(wide_df.columns), 2))
        for group_a, group_b in pairs:
            statistic, p_value = stats.wilcoxon(
                wide_df[group_a].to_numpy(dtype=float),
                wide_df[group_b].to_numpy(dtype=float),
                alternative="two-sided",
                zero_method="wilcox",
            )
            raw_p_values.append(float(p_value))
            diff = wide_df[group_a] - wide_df[group_b]
            pair_rows.append(
                {
                    "test_name": test.display_name,
                    "analysis_name": analysis_name,
                    "sheet_name": test.sheet_name,
                    "factor": test.factor,
                    "target": test.target,
                    "subject": test.subject,
                    "posthoc_method": "Wilcoxon signed-rank + Holm",
                    "group_a": str(group_a),
                    "group_b": str(group_b),
                    "statistic": float(statistic),
                    "raw_p": float(p_value),
                    "median_difference": float(np.median(diff)),
                    "n_used": int(len(diff)),
                    **repeated_summary,
                    "notes": "Pairwise Wilcoxon signed-rank tests with Holm correction.",
                }
            )
        _, adjusted_p_values, _, _ = multipletests(raw_p_values, alpha=alpha, method="holm")
        for row, adjusted_p in zip(pair_rows, adjusted_p_values):
            row["adjusted_p"] = float(adjusted_p)
            row["significant"] = bool(adjusted_p < alpha)
        return pd.DataFrame(pair_rows), "Post-hoc run: pairwise Wilcoxon signed-rank tests with Holm correction after significant Friedman test."

    return pd.DataFrame(), "No post-hoc procedure was executed."


def run_requested_analysis(
    test: AnalysisTest,
    subset: pd.DataFrame,
    alpha: float,
    levels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], str]:
    analysis_name = test.canonical_analysis_name
    base = analysis_result_row_base(test, alpha, levels)

    if analysis_name in SUBJECT_BASED_ANALYSES:
        aggregated_long, wide_df, repeated_summary = build_repeated_matrix(test, subset)
        if wide_df.empty:
            raise ValueError("No complete-case subjects remained for the requested paired/repeated analysis.")
        analysis_input_df = make_analysis_input_df(test, wide_df.reset_index(), "wide_complete_case")
    else:
        labels, arrays = build_independent_groups(subset, test.target)
        analysis_input_df = make_analysis_input_df(test, subset[[test.factor, "_group_label", test.target]].copy(), "long_grouped")
        repeated_summary = {}

    if analysis_name == "Paired t-test":
        col_a, col_b = list(wide_df.columns[:2])
        statistic, p_value = stats.ttest_rel(wide_df[col_a].to_numpy(dtype=float), wide_df[col_b].to_numpy(dtype=float))
        diff = wide_df[col_a] - wide_df[col_b]
        diff_shapiro = shapiro_result(diff.to_numpy(dtype=float), alpha)
        result_row = {
            **base,
            **repeated_summary,
            "statistic_name": "t",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "df": int(len(diff) - 1),
            "n_used": int(len(diff)),
            "level_a": col_a,
            "level_b": col_b,
            "mean_difference": float(diff.mean()),
            "difference_shapiro_w": diff_shapiro["w_stat"],
            "difference_shapiro_p": diff_shapiro["p_value"],
            "difference_normal": diff_shapiro["normal"],
            "significant": bool(p_value < alpha),
            "notes": "Paired t-test on complete-case subjects.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- levels: {col_a} vs {col_b}\n"
            f"- t={format_float(statistic)}, p={format_float(p_value)}, n_pairs={len(diff)}\n"
            f"- difference-score Shapiro p={format_float(diff_shapiro['p_value'])}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    if analysis_name == "Wilcoxon signed-rank":
        col_a, col_b = list(wide_df.columns[:2])
        statistic, p_value = stats.wilcoxon(
            wide_df[col_a].to_numpy(dtype=float),
            wide_df[col_b].to_numpy(dtype=float),
            alternative="two-sided",
            zero_method="wilcox",
        )
        diff = wide_df[col_a] - wide_df[col_b]
        result_row = {
            **base,
            **repeated_summary,
            "statistic_name": "W",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n_used": int(len(diff)),
            "level_a": col_a,
            "level_b": col_b,
            "median_difference": float(np.median(diff)),
            "significant": bool(p_value < alpha),
            "notes": "Wilcoxon signed-rank test on complete-case subjects.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- levels: {col_a} vs {col_b}\n"
            f"- W={format_float(statistic)}, p={format_float(p_value)}, n_pairs={len(diff)}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    if analysis_name == "Repeated-measures ANOVA":
        if test.subject is None:
            raise ValueError("Repeated-measures ANOVA requires a subject column.")
        long_df = wide_df.reset_index().melt(id_vars=[test.subject], var_name="within_level", value_name="analysis_value")
        model = AnovaRM(data=long_df, depvar="analysis_value", subject=test.subject, within=["within_level"])
        fit = model.fit()
        table = fit.anova_table.reset_index().rename(columns={"index": "effect"})
        row0 = table.iloc[0].to_dict()
        result_row = {
            **base,
            **repeated_summary,
            "statistic_name": "F",
            "statistic": float(row0.get("F Value", np.nan)),
            "p_value": float(row0.get("Pr > F", np.nan)),
            "df_num": float(row0.get("Num DF", np.nan)),
            "df_den": float(row0.get("Den DF", np.nan)),
            "n_used": int(wide_df.shape[0]),
            "significant": bool(float(row0.get("Pr > F", np.nan)) < alpha) if pd.notna(row0.get("Pr > F", np.nan)) else False,
            "notes": "Repeated-measures ANOVA on complete-case subjects.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- F={format_float(result_row['statistic'])}, p={format_float(result_row['p_value'])}, "
            f"subjects={wide_df.shape[0]}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    if analysis_name == "Friedman test":
        arrays = [wide_df[column].to_numpy(dtype=float) for column in wide_df.columns]
        statistic, p_value = stats.friedmanchisquare(*arrays)
        result_row = {
            **base,
            **repeated_summary,
            "statistic_name": "chi2",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "df": int(len(wide_df.columns) - 1),
            "n_used": int(wide_df.shape[0]),
            "significant": bool(p_value < alpha),
            "notes": "Friedman test on complete-case subjects.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- chi2={format_float(statistic)}, p={format_float(p_value)}, subjects={wide_df.shape[0]}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    labels, arrays = build_independent_groups(subset, test.target)
    if analysis_name == "Two-sample t-test (pooled)":
        statistic, p_value = stats.ttest_ind(arrays[0], arrays[1], equal_var=True)
        result_row = {
            **base,
            "statistic_name": "t",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n_used": int(sum(len(arr) for arr in arrays)),
            "level_a": labels[0],
            "level_b": labels[1],
            "mean_a": float(np.mean(arrays[0])),
            "mean_b": float(np.mean(arrays[1])),
            "significant": bool(p_value < alpha),
            "notes": "Independent two-sample t-test with pooled variance.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- levels: {labels[0]} vs {labels[1]}\n"
            f"- t={format_float(statistic)}, p={format_float(p_value)}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    if analysis_name == "Welch's t-test":
        statistic, p_value = stats.ttest_ind(arrays[0], arrays[1], equal_var=False)
        result_row = {
            **base,
            "statistic_name": "t",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n_used": int(sum(len(arr) for arr in arrays)),
            "level_a": labels[0],
            "level_b": labels[1],
            "mean_a": float(np.mean(arrays[0])),
            "mean_b": float(np.mean(arrays[1])),
            "significant": bool(p_value < alpha),
            "notes": "Independent Welch t-test.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- levels: {labels[0]} vs {labels[1]}\n"
            f"- t={format_float(statistic)}, p={format_float(p_value)}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    if analysis_name == "Mann-Whitney (Wilcoxon rank-sum)":
        statistic, p_value = stats.mannwhitneyu(arrays[0], arrays[1], alternative="two-sided")
        result_row = {
            **base,
            "statistic_name": "U",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n_used": int(sum(len(arr) for arr in arrays)),
            "level_a": labels[0],
            "level_b": labels[1],
            "median_a": float(np.median(arrays[0])),
            "median_b": float(np.median(arrays[1])),
            "significant": bool(p_value < alpha),
            "notes": "Mann-Whitney rank-sum test.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- levels: {labels[0]} vs {labels[1]}\n"
            f"- U={format_float(statistic)}, p={format_float(p_value)}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    if analysis_name == "One-way ANOVA":
        statistic, p_value = stats.f_oneway(*arrays)
        result_row = {
            **base,
            "statistic_name": "F",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n_used": int(sum(len(arr) for arr in arrays)),
            "n_groups": int(len(arrays)),
            "significant": bool(p_value < alpha),
            "notes": "Classical one-way ANOVA.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- F={format_float(statistic)}, p={format_float(p_value)}, groups={len(arrays)}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    if analysis_name == "Welch ANOVA":
        result = anova_oneway(subset[test.target].to_numpy(dtype=float), groups=subset["_group_label"].astype(str).to_numpy(), use_var="unequal")
        df_tuple = getattr(result, "df", (np.nan, np.nan))
        if isinstance(df_tuple, tuple) and len(df_tuple) == 2:
            df_num, df_den = df_tuple
        else:
            df_num, df_den = np.nan, np.nan
        statistic = getattr(result, "statistic", np.nan)
        p_value = getattr(result, "pvalue", np.nan)
        result_row = {
            **base,
            "statistic_name": "F",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "df_num": float(df_num),
            "df_den": float(df_den),
            "n_used": int(len(subset)),
            "n_groups": int(len(labels)),
            "significant": bool(float(p_value) < alpha) if pd.notna(p_value) else False,
            "notes": "Welch one-way ANOVA.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- F={format_float(statistic)}, p={format_float(p_value)}, groups={len(labels)}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    if analysis_name == "Kruskal-Wallis":
        statistic, p_value = stats.kruskal(*arrays)
        result_row = {
            **base,
            "statistic_name": "H",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "df": int(len(arrays) - 1),
            "n_used": int(sum(len(arr) for arr in arrays)),
            "n_groups": int(len(arrays)),
            "significant": bool(p_value < alpha),
            "notes": "Kruskal-Wallis omnibus test.",
        }
        text = (
            f"Requested analysis: {analysis_name}\n"
            f"- H={format_float(statistic)}, p={format_float(p_value)}, groups={len(arrays)}"
        )
        return pd.DataFrame([result_row]), analysis_input_df, result_row, text

    raise ValueError(f"Analysis not implemented: {analysis_name}")


def format_analysis_result_table(analysis_results_df: pd.DataFrame) -> list[str]:
    if analysis_results_df.empty:
        return ["ANALYSIS RESULT", "- No analysis result."]

    display_df = analysis_results_df.copy()
    for column in display_df.columns:
        if pd.api.types.is_bool_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: "True" if value else "False")
        elif pd.api.types.is_numeric_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: format_float(value) if pd.notna(value) else "")
    return ["ANALYSIS RESULT", display_df.to_string(index=False)]


def format_posthoc_result_table(posthoc_results_df: pd.DataFrame) -> list[str]:
    if posthoc_results_df.empty:
        return ["POST-HOC RESULT", "- No post-hoc result."]

    display_df = posthoc_results_df.copy()
    for column in display_df.columns:
        if pd.api.types.is_bool_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: "True" if value else "False")
        elif pd.api.types.is_numeric_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: format_float(value) if pd.notna(value) else "")
    return ["POST-HOC RESULT", display_df.to_string(index=False)]


def analyze_test(df: pd.DataFrame, test: AnalysisTest, alpha: float) -> DetailedTestResult:
    subset, rows_before_drop, realized_levels, original_non_null_count = prepare_subset(df, test)
    group_df = build_group_diagnostics(subset, factor_label_col="_group_label", target=test.target, alpha=alpha)

    continuous_verdict, continuous_detail, continuous_summary = assess_continuous_dv(
        values=to_numeric(df[test.target]),
        original_non_null_count=original_non_null_count,
    )
    independence_verdict, independence_detail, independence_summary = assess_independence(df, test.subject)
    normality_verdict, normality_detail, normality_summary = assess_normality(subset, "_group_label", test.target, group_df, alpha)
    variance_verdict, variance_detail, variance_summary = assess_variance_homogeneity(group_df, subset, "_group_label", test.target, alpha)
    outlier_verdict, outlier_detail, outlier_summary = assess_outliers(group_df)

    analysis_results_df, analysis_input_df, result_row, analysis_text = run_requested_analysis(
        test=test,
        subset=subset,
        alpha=alpha,
        levels=realized_levels,
    )
    posthoc_results_df, posthoc_text = run_posthoc_analysis(
        test=test,
        subset=subset,
        alpha=alpha,
        levels=realized_levels,
        analysis_result_row=result_row,
    )
    next_step = suggest_next_step(test.canonical_analysis_name, normality_verdict, variance_verdict, outlier_verdict)

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
        f"Requested analysis: {test.canonical_analysis_name}",
        f"Rows analyzed: {len(subset)}",
        f"Groups analyzed: {subset['_group_label'].nunique()}",
        f"Levels used: {', '.join(realized_levels)}",
        f"Alpha: {alpha}",
        *format_assumption_lines(answers),
        "NEXT STEP",
        f"- {next_step}",
        "POST-HOC NOTE",
        f"- {posthoc_text}",
        *format_analysis_result_table(analysis_results_df),
        *format_posthoc_result_table(posthoc_results_df),
        *format_group_table(group_df),
    ]

    assumptions_df = build_assumptions_dataframe(test, answers)
    group_diagnostics_df = group_df.copy()
    if not group_diagnostics_df.empty:
        group_diagnostics_df.insert(0, "target", test.target)
        group_diagnostics_df.insert(0, "factor", test.factor)
        group_diagnostics_df.insert(0, "subject", test.subject)
        group_diagnostics_df.insert(0, "sheet_name", test.sheet_name)
        group_diagnostics_df.insert(0, "analysis_name", test.canonical_analysis_name)
        group_diagnostics_df.insert(0, "test_name", test.display_name)

    summary_row = {
        "test_name": test.display_name,
        "analysis_name": test.canonical_analysis_name,
        "sheet_name": test.sheet_name,
        "factor": test.factor,
        "target": test.target,
        "subject": test.subject,
        "levels": " | ".join(realized_levels),
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
        "analysis_statistic_name": result_row.get("statistic_name"),
        "analysis_statistic": result_row.get("statistic"),
        "analysis_p_value": result_row.get("p_value"),
        "analysis_significant": result_row.get("significant"),
        "posthoc_run": bool(not posthoc_results_df.empty),
        "posthoc_rows": int(len(posthoc_results_df)),
        "posthoc_note": posthoc_text,
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
        analysis_results_df=analysis_results_df,
        analysis_input_df=analysis_input_df,
        posthoc_results_df=posthoc_results_df,
    )


def save_detailed_results(output_file: Path, results: list[DetailedTestResult], error_records: list[dict[str, object]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([result.summary_row for result in results])
    assumptions_df = pd.concat([result.assumptions_df for result in results], ignore_index=True) if results else pd.DataFrame()
    group_diagnostics_df = pd.concat([result.group_diagnostics_df for result in results], ignore_index=True) if results else pd.DataFrame()
    analyzed_rows_df = pd.concat([result.analyzed_rows_df for result in results], ignore_index=True) if results else pd.DataFrame()
    analysis_results_df = pd.concat([result.analysis_results_df for result in results], ignore_index=True) if results else pd.DataFrame()
    analysis_inputs_df = pd.concat([result.analysis_input_df for result in results], ignore_index=True) if results else pd.DataFrame()
    posthoc_results_df = pd.concat([result.posthoc_results_df for result in results], ignore_index=True) if results else pd.DataFrame()
    errors_df = pd.DataFrame(error_records)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        assumptions_df.to_excel(writer, sheet_name="assumptions", index=False)
        group_diagnostics_df.to_excel(writer, sheet_name="group_diagnostics", index=False)
        analyzed_rows_df.to_excel(writer, sheet_name="analyzed_rows", index=False)
        analysis_results_df.to_excel(writer, sheet_name="analysis_results", index=False)
        analysis_inputs_df.to_excel(writer, sheet_name="analysis_inputs", index=False)
        posthoc_results_df.to_excel(writer, sheet_name="posthoc_results", index=False)
        errors_df.to_excel(writer, sheet_name="errors", index=False)


def main() -> None:
    args = parse_args()
    tests = build_tests()

    report_sections: list[str] = []
    detailed_results: list[DetailedTestResult] = []
    error_records: list[dict[str, object]] = []

    if args.describe_input:
        report_sections.append(describe_workbook(args.input))

    workbook = pd.ExcelFile(args.input)
    sheet_cache: dict[str, pd.DataFrame] = {}

    for test in tests:
        try:
            _ = test.canonical_analysis_name
            if test.sheet_name not in workbook.sheet_names:
                raise ValueError(f"Sheet '{test.sheet_name}' does not exist. Available sheets: {workbook.sheet_names}")
            if test.sheet_name not in sheet_cache:
                sheet_cache[test.sheet_name] = load_sheet(args.input, test.sheet_name)
            result = analyze_test(sheet_cache[test.sheet_name], test=test, alpha=args.alpha)
            detailed_results.append(result)
            report_sections.append(result.text_report)
        except Exception as exc:
            error_records.append(
                {
                    "test_name": test.name or "",
                    "analysis_name": test.analysis_name,
                    "sheet_name": test.sheet_name,
                    "factor": test.factor,
                    "target": test.target,
                    "subject": test.subject,
                    "levels": " | ".join(test.levels) if test.levels else "",
                    "error": str(exc),
                }
            )
            display_name = test.name or f"{test.analysis_name} | {test.sheet_name} | factor={test.factor} | target={test.target}"
            report_sections.append("\n".join(["=" * 100, f"TEST: {display_name}", f"ERROR: {exc}"]))

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
