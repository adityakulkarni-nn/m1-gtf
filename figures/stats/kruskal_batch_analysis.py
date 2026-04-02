from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "master_error.xlsx"
DEFAULT_OUTPUT = SCRIPT_DIR / "KW"/ "kruskal_batch_results.xlsx"
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class AnalysisTest:
    test_name: str
    sheet_name: str
    factor: str
    target_col: str


@dataclass(frozen=True)
class TestResult:
    overall_shapiro: pd.DataFrame
    group_shapiro: pd.DataFrame
    shapiro_summary: pd.DataFrame
    kruskal_result: pd.DataFrame
    pairwise_results: pd.DataFrame


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing {label} columns: {missing}")


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


def compute_shapiro_results(
    df: pd.DataFrame,
    test: AnalysisTest,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_values = to_numeric(df[test.target_col]).dropna().to_numpy(dtype=float)
    overall_df = pd.DataFrame(
        [
            {
                "test_name": test.test_name,
                "sheet_name": test.sheet_name,
                "factor": test.factor,
                "target_col": test.target_col,
                "scope": "overall",
                "level": "ALL",
                **shapiro_record(target_values, alpha),
            }
        ]
    )

    subset = df[[test.factor, test.target_col]].copy()
    subset[test.target_col] = to_numeric(subset[test.target_col])
    subset = subset.dropna(subset=[test.factor, test.target_col])

    group_records: list[dict[str, object]] = []
    for level, group_df in subset.groupby(test.factor, dropna=True, sort=True):
        values = group_df[test.target_col].to_numpy(dtype=float)
        group_records.append(
            {
                "test_name": test.test_name,
                "sheet_name": test.sheet_name,
                "factor": test.factor,
                "target_col": test.target_col,
                "scope": "group",
                "level": format_group_value(level),
                **shapiro_record(values, alpha),
            }
        )

    group_df = pd.DataFrame.from_records(group_records)
    if group_df.empty:
        summary_df = pd.DataFrame(
            [
                {
                    "test_name": test.test_name,
                    "sheet_name": test.sheet_name,
                    "factor": test.factor,
                    "target_col": test.target_col,
                    "groups_evaluated": 0,
                    "groups_tested": 0,
                    "groups_non_normal": 0,
                    "groups_insufficient_n": 0,
                    "all_tested_groups_normal": True,
                }
            ]
        )
        return overall_df, group_df, summary_df

    tested = group_df["test_ran"] == True
    non_normal = group_df["normal_at_alpha"] == False
    summary_df = pd.DataFrame(
        [
            {
                "test_name": test.test_name,
                "sheet_name": test.sheet_name,
                "factor": test.factor,
                "target_col": test.target_col,
                "groups_evaluated": int(len(group_df)),
                "groups_tested": int(tested.sum()),
                "groups_non_normal": int((tested & non_normal).sum()),
                "groups_insufficient_n": int((group_df["test_ran"] == False).sum()),
                "all_tested_groups_normal": bool(group_df.loc[tested, "normal_at_alpha"].all()) if tested.any() else True,
            }
        ]
    )
    return overall_df, group_df, summary_df


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


def describe_pairwise_direction(group_a: str, group_b: str, median_a: float, median_b: float) -> tuple[str, str]:
    if median_a > median_b:
        return f"{group_a} > {group_b}", f"Dunn's test with Holm correction showed that {group_a} > {group_b}"
    if median_b > median_a:
        return f"{group_b} > {group_a}", f"Dunn's test with Holm correction showed that {group_b} > {group_a}"
    return (
        f"{group_a} = {group_b}",
        f"Dunn's test with Holm correction showed a difference between {group_a} and {group_b}, but their medians were equal",
    )


def run_dunn_posthoc(
    subset: pd.DataFrame,
    test: AnalysisTest,
    group_order: list[str],
    group_map: dict[str, np.ndarray],
    alpha: float,
) -> pd.DataFrame:
    analysis_df = subset[[test.factor, test.target_col]].copy()
    analysis_df["_group_label"] = analysis_df[test.factor].map(format_group_value)
    analysis_df = analysis_df.dropna(subset=["_group_label", test.target_col])

    raw_matrix = sp.posthoc_dunn(
        analysis_df,
        val_col=test.target_col,
        group_col="_group_label",
        p_adjust=None,
        sort=False,
    )
    holm_matrix = sp.posthoc_dunn(
        analysis_df,
        val_col=test.target_col,
        group_col="_group_label",
        p_adjust="holm",
        sort=False,
    )

    pairwise_rows: list[dict[str, object]] = []
    for index, group_a in enumerate(group_order[:-1]):
        values_a = group_map[group_a]
        for group_b in group_order[index + 1 :]:
            values_b = group_map[group_b]
            raw_p = float(raw_matrix.loc[group_a, group_b])
            holm_p = float(holm_matrix.loc[group_a, group_b])
            comparison, interpretation = describe_pairwise_direction(
                group_a=group_a,
                group_b=group_b,
                median_a=float(np.median(values_a)),
                median_b=float(np.median(values_b)),
            )
            pairwise_rows.append(
                {
                    "test_name": test.test_name,
                    "sheet_name": test.sheet_name,
                    "factor": test.factor,
                    "target_col": test.target_col,
                    "posthoc_method": "Dunn",
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": int(values_a.size),
                    "n_b": int(values_b.size),
                    "median_a": float(np.median(values_a)),
                    "median_b": float(np.median(values_b)),
                    "comparison": comparison,
                    "raw_p": raw_p,
                    "holm_p": holm_p,
                    "significant": bool(holm_p < alpha),
                    "interpretation": interpretation,
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    if pairwise_df.empty:
        return pairwise_df

    return pairwise_df.sort_values(
        by=["test_name", "holm_p", "raw_p", "group_a", "group_b"]
    ).reset_index(drop=True)


def run_kruskal(
    df: pd.DataFrame,
    test: AnalysisTest,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[[test.factor, test.target_col]].copy()
    subset[test.target_col] = to_numeric(subset[test.target_col])
    subset = subset.dropna(subset=[test.factor, test.target_col])

    grouped_values: list[np.ndarray] = []
    group_order: list[str] = []
    group_map: dict[str, np.ndarray] = {}
    group_sizes: dict[str, int] = {}
    group_medians: dict[str, float] = {}

    for level, group_df in subset.groupby(test.factor, dropna=True, sort=True):
        values = group_df[test.target_col].to_numpy(dtype=float)
        if values.size == 0:
            continue
        group_name = format_group_value(level)
        grouped_values.append(values)
        group_order.append(group_name)
        group_map[group_name] = values
        group_sizes[group_name] = int(values.size)
        group_medians[group_name] = float(np.median(values))

    result_row: dict[str, object] = {
        "test_name": test.test_name,
        "sheet_name": test.sheet_name,
        "factor": test.factor,
        "target_col": test.target_col,
        "alpha": alpha,
        "n_groups": len(grouped_values),
        "df": max(0, len(grouped_values) - 1),
        "total_n": int(sum(group_sizes.values())),
        "group_sizes": "; ".join(f"{group}={size}" for group, size in group_sizes.items()),
        "group_medians": "; ".join(f"{group}={group_medians[group]:.4f}" for group in group_order),
        "h_stat": np.nan,
        "p_value": np.nan,
        "epsilon_squared": np.nan,
        "effect_size": "undetermined",
        "significant": False,
        "pairwise_run": False,
        "notes": "",
    }

    if len(grouped_values) < 2:
        result_df = pd.DataFrame([result_row])
        result_df.loc[0, "notes"] = "At least 2 non-empty groups are required for Kruskal-Wallis."
        return result_df, pd.DataFrame()

    try:
        h_stat, p_value = stats.kruskal(*grouped_values)
    except ValueError as exc:
        result_df = pd.DataFrame([result_row])
        result_df.loc[0, "notes"] = f"Kruskal-Wallis failed: {exc}"
        return result_df, pd.DataFrame()

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

    pairwise_df = run_dunn_posthoc(
        subset=subset,
        test=test,
        group_order=group_order,
        group_map=group_map,
        alpha=alpha,
    )
    if not pairwise_df.empty:
        result_row["pairwise_run"] = True
        result_row["notes"] = "Dunn's post-hoc comparisons with Holm correction were added for this test."

    return pd.DataFrame([result_row]), pairwise_df


def build_posthoc_summary(kruskal_results_df: pd.DataFrame, pairwise_results_df: pd.DataFrame) -> pd.DataFrame:
    if kruskal_results_df.empty:
        return pd.DataFrame(columns=["test_name", "summary"])

    summary_rows: list[dict[str, object]] = []
    for _, kruskal_row in kruskal_results_df.iterrows():
        matching_pairs = pairwise_results_df[
            (pairwise_results_df["test_name"] == kruskal_row["test_name"])
            & (pairwise_results_df["significant"] == True)
        ].copy()

        if matching_pairs.empty:
            if bool(kruskal_row["significant"]):
                summary = (
                    "Kruskal-Wallis was significant, but no Dunn's test contrast remained significant after Holm correction."
                )
            else:
                summary = (
                    "Dunn's test with Holm correction was run, but no pairwise contrast was significant."
                )
        else:
            contrast_text = "; ".join(
                f"{pair['comparison']} (Holm p = {pair['holm_p']:.4f})"
                for _, pair in matching_pairs.iterrows()
            )
            summary = f"Dunn's test with Holm correction found: {contrast_text}."

        summary_rows.append(
            {
                "test_name": kruskal_row["test_name"],
                "sheet_name": kruskal_row["sheet_name"],
                "factor": kruskal_row["factor"],
                "target_col": kruskal_row["target_col"],
                "summary": summary,
            }
        )

    return pd.DataFrame(summary_rows)


def run_test(df: pd.DataFrame, test: AnalysisTest, alpha: float) -> TestResult:
    require_columns(df, [test.factor, test.target_col], f"test '{test.test_name}'")
    overall_shapiro, group_shapiro, shapiro_summary = compute_shapiro_results(df=df, test=test, alpha=alpha)
    kruskal_result, pairwise_results = run_kruskal(df=df, test=test, alpha=alpha)
    return TestResult(
        overall_shapiro=overall_shapiro,
        group_shapiro=group_shapiro,
        shapiro_summary=shapiro_summary,
        kruskal_result=kruskal_result,
        pairwise_results=pairwise_results,
    )


def write_results(
    output_file: Path,
    tests_df: pd.DataFrame,
    overall_shapiro_df: pd.DataFrame,
    group_shapiro_df: pd.DataFrame,
    shapiro_summary_df: pd.DataFrame,
    kruskal_results_df: pd.DataFrame,
    pairwise_results_df: pd.DataFrame,
    posthoc_summary_df: pd.DataFrame,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        tests_df.to_excel(writer, sheet_name="tests", index=False)
        overall_shapiro_df.to_excel(writer, sheet_name="shapiro_overall", index=False)
        group_shapiro_df.to_excel(writer, sheet_name="shapiro_by_group", index=False)
        shapiro_summary_df.to_excel(writer, sheet_name="shapiro_summary", index=False)
        kruskal_results_df.to_excel(writer, sheet_name="kruskal_results", index=False)
        pairwise_results_df.to_excel(writer, sheet_name="pairwise_results", index=False)
        posthoc_summary_df.to_excel(writer, sheet_name="posthoc_summary", index=False)


def print_summary(output_file: Path, kruskal_results_df: pd.DataFrame, posthoc_summary_df: pd.DataFrame) -> None:
    print("=" * 78)
    print("BATCH SHAPIRO-WILK AND KRUSKAL-WALLIS ANALYSIS")
    print("=" * 78)
    print(kruskal_results_df[
        [
            "test_name",
            "sheet_name",
            "factor",
            "target_col",
            "n_groups",
            "h_stat",
            "p_value",
            "epsilon_squared",
            "effect_size",
            "significant",
        ]
    ].to_string(index=False))
    if not posthoc_summary_df.empty:
        print("\nPost-hoc summary:")
        for _, row in posthoc_summary_df.iterrows():
            print(f"- {row['test_name']}: {row['summary']}")
    print(f"\nExcel results: {output_file}")


def main() -> None:
    input_file = DEFAULT_INPUT
    output_file = DEFAULT_OUTPUT
    alpha = DEFAULT_ALPHA

    tests = [
        AnalysisTest(
            test_name="adjusted_total_error_by_operator",
            sheet_name="adjusted_position",
            factor="operator",
            target_col="total_error",
        ),
        AnalysisTest(
            test_name="adjusted_total_error_by_side",
            sheet_name="adjusted_position",
            factor="side",
            target_col="total_error",
        ),
        AnalysisTest(
            test_name="adjusted_total_error_by_arc",
            sheet_name="adjusted_position",
            factor="arc",
            target_col="total_error",
        ),
        AnalysisTest(
            test_name="adjusted_total_error_by_collar",
            sheet_name="adjusted_position",
            factor="collar",
            target_col="total_error",
        ),
        # Add more tests here, for example:
        # AnalysisTest(
        #     test_name="adjusted_total_error_by_arc",
        #     sheet_name="adjusted_position",
        #     factor="arc",
        #     target_col="total_error",
        # ),
        AnalysisTest(
            test_name="adapter_total_error_by_operator",
            sheet_name="adapter",
            factor="operator",
            target_col="total_error",
        ),
        AnalysisTest(
            test_name="adapter_total_error_by_side",
            sheet_name="adapter",
            factor="side",
            target_col="total_error",
        ),
        AnalysisTest(
            test_name="adapter_total_error_by_arc",
            sheet_name="adapter",
            factor="arc",
            target_col="total_error",
        ),
        AnalysisTest(
            test_name="adapter_total_error_by_collar",
            sheet_name="adapter",
            factor="collar",
            target_col="total_error",
        ),
        # Comparison adapter vs adjusted for total error:
        AnalysisTest(
            test_name="adapter_vs_adjusted_total_error",
            sheet_name="comparison",
            factor="method",
            target_col="total_error",
        ),
    ]

    if not input_file.exists():
        raise FileNotFoundError(f"Workbook not found: {input_file}")

    workbook = pd.ExcelFile(input_file)
    sheet_cache: dict[str, pd.DataFrame] = {}

    test_rows: list[dict[str, object]] = []
    overall_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    kruskal_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []

    for test in tests:
        if test.sheet_name not in workbook.sheet_names:
            raise KeyError(f"Sheet '{test.sheet_name}' not found. Available sheets: {workbook.sheet_names}")

        if test.sheet_name not in sheet_cache:
            sheet_df = workbook.parse(test.sheet_name).copy()
            sheet_df.columns = [str(column).strip() for column in sheet_df.columns]
            sheet_cache[test.sheet_name] = sheet_df

        result = run_test(df=sheet_cache[test.sheet_name], test=test, alpha=alpha)
        test_rows.append(
            {
                "test_name": test.test_name,
                "sheet_name": test.sheet_name,
                "factor": test.factor,
                "target_col": test.target_col,
                "alpha": alpha,
            }
        )
        overall_frames.append(result.overall_shapiro)
        group_frames.append(result.group_shapiro)
        summary_frames.append(result.shapiro_summary)
        kruskal_frames.append(result.kruskal_result)
        if not result.pairwise_results.empty:
            pairwise_frames.append(result.pairwise_results)

    tests_df = pd.DataFrame(test_rows)
    overall_shapiro_df = pd.concat(overall_frames, ignore_index=True) if overall_frames else pd.DataFrame()
    group_shapiro_df = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    shapiro_summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    kruskal_results_df = pd.concat(kruskal_frames, ignore_index=True) if kruskal_frames else pd.DataFrame()
    pairwise_results_df = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame(
        columns=[
            "test_name",
            "sheet_name",
            "factor",
            "target_col",
            "posthoc_method",
            "group_a",
            "group_b",
            "n_a",
            "n_b",
            "median_a",
            "median_b",
            "comparison",
            "raw_p",
            "holm_p",
            "significant",
            "interpretation",
        ]
    )
    posthoc_summary_df = build_posthoc_summary(kruskal_results_df=kruskal_results_df, pairwise_results_df=pairwise_results_df)

    write_results(
        output_file=output_file,
        tests_df=tests_df,
        overall_shapiro_df=overall_shapiro_df,
        group_shapiro_df=group_shapiro_df,
        shapiro_summary_df=shapiro_summary_df,
        kruskal_results_df=kruskal_results_df,
        pairwise_results_df=pairwise_results_df,
        posthoc_summary_df=posthoc_summary_df,
    )
    print_summary(
        output_file=output_file,
        kruskal_results_df=kruskal_results_df,
        posthoc_summary_df=posthoc_summary_df,
    )


if __name__ == "__main__":
    main()
