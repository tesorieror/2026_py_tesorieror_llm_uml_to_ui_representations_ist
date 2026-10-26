from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILENAME = Path(
    "experiment/analysis/tables/"
    "primary_element_weighted_contrasts.csv"
)

OUTPUT_PATH = Path(
    "experiment/analysis/tables"
)


OUTCOME_LEVELS = [
    "attribute_coverage",
    "primitive_type_fidelity",
    "enumeration_literal_coverage",
    "enumeration_cardinality_fidelity",
    "attribute_nonempty_label_coverage",
    "association_end_coverage",
    "association_cardinality_fidelity",
]


MODEL_LEVELS = [
    "llama3.1:8b",
    "mistral:7b-instruct",
    "qwen2.5:7b-instruct",
]


COMPARISON_LEVELS = [
    "json",
    "xsd",
    "sql",
    "txt",
]


EXPECTED_ROWS = (
    len(OUTCOME_LEVELS)
    * len(MODEL_LEVELS)
    * len(COMPARISON_LEVELS)
)

def load_primary_contrasts():
    if not INPUT_FILENAME.is_file():
        raise FileNotFoundError(
            f"Primary contrast table not found: "
            f"{INPUT_FILENAME}"
        )

    return pd.read_csv(
        INPUT_FILENAME
    )

def validate_primary_contrasts(
    dataframe,
):
    required_columns = {
        "outcome",
        "model",
        "reference",
        "comparison",
        "n_paired_umls",
        "paired_umls",
        "analysis_status",
        "element_weighted_difference",
        "element_bb_mean",
        "element_bb_median",
        "element_bb_q025",
        "element_bb_q975",
        "element_bb_probability_gt_0",
        "element_bb_probability_lt_0",
        "element_bb_probability_eq_0",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing primary contrast columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(dataframe)}."
        )

    if dataframe[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate primary contrast rows."
        )

    if not (
        dataframe["reference"] == "uml"
    ).all():
        raise ValueError(
            "Unexpected reference representation."
        )

    if set(
        dataframe["outcome"]
    ) != set(OUTCOME_LEVELS):
        raise ValueError(
            "Unexpected outcome levels."
        )

    if set(
        dataframe["model"]
    ) != set(MODEL_LEVELS):
        raise ValueError(
            "Unexpected model levels."
        )

    if set(
        dataframe["comparison"]
    ) != set(COMPARISON_LEVELS):
        raise ValueError(
            "Unexpected comparison levels."
        )

def validate_primary_contrasts(
    dataframe,
):
    required_columns = {
        "outcome",
        "model",
        "reference",
        "comparison",
        "n_paired_umls",
        "paired_umls",
        "analysis_status",
        "element_weighted_difference",
        "element_bb_mean",
        "element_bb_median",
        "element_bb_q025",
        "element_bb_q975",
        "element_bb_probability_gt_0",
        "element_bb_probability_lt_0",
        "element_bb_probability_eq_0",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing primary contrast columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(dataframe)}."
        )

    if dataframe[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate primary contrast rows."
        )

    if not (
        dataframe["reference"] == "uml"
    ).all():
        raise ValueError(
            "Unexpected reference representation."
        )

    if set(
        dataframe["outcome"]
    ) != set(OUTCOME_LEVELS):
        raise ValueError(
            "Unexpected outcome levels."
        )

    if set(
        dataframe["model"]
    ) != set(MODEL_LEVELS):
        raise ValueError(
            "Unexpected model levels."
        )

    if set(
        dataframe["comparison"]
    ) != set(COMPARISON_LEVELS):
        raise ValueError(
            "Unexpected comparison levels."
        )

def format_float(
    value,
    digits=3,
):
    if pd.isna(value):
        return "NA"

    return f"{value:.{digits}f}"

def format_contrast_cell(
    row,
):
    status = row[
        "analysis_status"
    ]

    n_paired = int(
        row["n_paired_umls"]
    )

    if status == "not_estimable":
        return (
            "not estimable "
            f"(n={n_paired})"
        )

    if status == "descriptive_only":
        difference = format_float(
            row[
                "element_weighted_difference"
            ]
        )

        return (
            f"Δ={difference}; "
            f"descriptive only; "
            f"n={n_paired}"
        )

    if status != "bayesian_bootstrap":
        raise ValueError(
            f"Unexpected analysis_status: "
            f"{status}"
        )

    mean = format_float(
        row["element_bb_mean"]
    )

    lower = format_float(
        row["element_bb_q025"]
    )

    upper = format_float(
        row["element_bb_q975"]
    )

    return (
        f"Δ={mean} "
        f"[{lower}, {upper}]; "
        f"n={n_paired}"
    )

def build_summary_matrix(
    dataframe,
):
    rows = []

    for outcome_name in OUTCOME_LEVELS:
        for model_name in MODEL_LEVELS:
            row = {
                "outcome": outcome_name,
                "model": model_name,
            }

            subset = dataframe[
                (
                    dataframe["outcome"]
                    == outcome_name
                )
                & (
                    dataframe["model"]
                    == model_name
                )
            ]

            if len(subset) != len(
                COMPARISON_LEVELS
            ):
                raise ValueError(
                    "Unexpected number of "
                    "comparisons for "
                    f"{outcome_name} / "
                    f"{model_name}."
                )

            for comparison_name in (
                COMPARISON_LEVELS
            ):
                comparison_rows = subset[
                    subset["comparison"]
                    == comparison_name
                ]

                if len(
                    comparison_rows
                ) != 1:
                    raise ValueError(
                        "Expected exactly one row "
                        f"for {outcome_name} / "
                        f"{model_name} / "
                        f"{comparison_name}."
                    )

                comparison_row = (
                    comparison_rows.iloc[0]
                )

                row[
                    comparison_name
                ] = format_contrast_cell(
                    comparison_row
                )

            rows.append(
                row
            )

    result = pd.DataFrame(
        rows
    )

    expected_summary_rows = (
        len(OUTCOME_LEVELS)
        * len(MODEL_LEVELS)
    )

    if len(result) != expected_summary_rows:
        raise ValueError(
            "Unexpected summary-matrix size."
        )

    return result

def build_compact_long_table(
    dataframe,
):
    columns = [
        "outcome",
        "model",
        "comparison",
        "n_paired_umls",
        "paired_umls",
        "analysis_status",
        "element_weighted_difference",
        "element_bb_mean",
        "element_bb_median",
        "element_bb_q025",
        "element_bb_q975",
        "element_bb_probability_gt_0",
        "element_bb_probability_lt_0",
        "element_bb_probability_eq_0",
    ]

    result = dataframe[
        columns
    ].copy()

    return result

def write_table(
    dataframe,
    filename,
):
    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_filename = (
        OUTPUT_PATH
        / filename
    )

    dataframe.to_csv(
        output_filename,
        index=False,
    )

    return output_filename

def validate_summary_matrix(
    matrix,
):
    expected_rows = (
        len(OUTCOME_LEVELS)
        * len(MODEL_LEVELS)
    )

    if len(matrix) != expected_rows:
        raise ValueError(
            "Unexpected number of "
            "summary-matrix rows."
        )

    if matrix[
        [
            "outcome",
            "model",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate summary-matrix rows."
        )

    expected_columns = {
        "outcome",
        "model",
        *COMPARISON_LEVELS,
    }

    if set(
        matrix.columns
    ) != expected_columns:
        raise ValueError(
            "Unexpected summary-matrix columns."
        )

def main():
    contrasts = (
        load_primary_contrasts()
    )

    validate_primary_contrasts(
        contrasts
    )

    matrix = build_summary_matrix(
        contrasts
    )

    compact_long = (
        build_compact_long_table(
            contrasts
        )
    )

    validate_summary_matrix(
        matrix
    )

    matrix_filename = write_table(
        matrix,
        "primary_contrast_summary_matrix.csv",
    )

    long_filename = write_table(
        compact_long,
        "primary_contrast_summary_long.csv",
    )

    print(
        "Primary cell-level contrast summaries "
        "prepared and validated successfully."
    )

    print(
        f"matrix_rows: {len(matrix)}"
    )

    print(
        f"long_rows: {len(compact_long)}"
    )

    print()

    print(
        "Summary matrix:"
    )

    print(
        matrix.to_string(
            index=False
        )
    )

    print()

    print(
        f"saved: {matrix_filename}"
    )

    print(
        f"saved: {long_filename}"
    )


if __name__ == "__main__":
    main()

    