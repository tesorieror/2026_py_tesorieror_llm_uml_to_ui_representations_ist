from pathlib import Path

import pandas as pd


INPUT_FILENAME = Path(
    "experiment/analysis/tables/"
    "primary_contrast_summary_long.csv"
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


EXPECTED_ROWS = 84
EXPECTED_TABLE_ROWS = 21

def load_data():
    if not INPUT_FILENAME.is_file():
        raise FileNotFoundError(
            f"Input table not found: "
            f"{INPUT_FILENAME}"
        )

    return pd.read_csv(
        INPUT_FILENAME
    )

def validate_data(
    dataframe,
):
    required_columns = {
        "outcome",
        "model",
        "comparison",
        "n_paired_umls",
        "analysis_status",
        "element_weighted_difference",
        "element_bb_mean",
        "element_bb_q025",
        "element_bb_q975",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
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
            "Duplicate primary-result rows."
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

def format_signed(
    value,
    digits=3,
):
    if pd.isna(value):
        return "NA"

    return f"{value:+.{digits}f}"    

def format_interval(
    lower,
    upper,
    digits=3,
):
    if (
        pd.isna(lower)
        or pd.isna(upper)
    ):
        return "NA"

    return (
        "["
        f"{lower:+.{digits}f}, "
        f"{upper:+.{digits}f}"
        "]"
    )

def format_result_cell(
    row,
):
    status = row[
        "analysis_status"
    ]

    n_paired = int(
        row["n_paired_umls"]
    )

    if status == "not_estimable":
        return "NE"

    if status == "descriptive_only":
        difference = format_signed(
            row[
                "element_weighted_difference"
            ]
        )

        return (
            f"{difference} "
            f"(descriptive; n={n_paired})"
        )

    if status != "bayesian_bootstrap":
        raise ValueError(
            f"Unexpected analysis_status: "
            f"{status}"
        )

    mean = format_signed(
        row[
            "element_bb_mean"
        ]
    )

    interval = format_interval(
        row[
            "element_bb_q025"
        ],
        row[
            "element_bb_q975"
        ],
    )

    return (
        f"{mean} {interval}"
    )

def build_results_table(
    dataframe,
):
    rows = []

    for outcome_name in OUTCOME_LEVELS:
        for model_name in MODEL_LEVELS:
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
                    "representations for "
                    f"{outcome_name} / "
                    f"{model_name}."
                )

            row = {
                "outcome": outcome_name,
                "model": model_name,
            }

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
                        "Expected exactly one "
                        "contrast for "
                        f"{outcome_name} / "
                        f"{model_name} / "
                        f"{comparison_name}."
                    )

                row[
                    comparison_name
                ] = format_result_cell(
                    comparison_rows.iloc[0]
                )

            rows.append(
                row
            )

    result = pd.DataFrame(
        rows
    )

    return result

def build_support_table(
    dataframe,
):
    rows = []

    for outcome_name in OUTCOME_LEVELS:
        for model_name in MODEL_LEVELS:
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

            row = {
                "outcome": outcome_name,
                "model": model_name,
            }

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
                        "Missing support row."
                    )

                row[
                    comparison_name
                ] = int(
                    comparison_rows
                    .iloc[0][
                        "n_paired_umls"
                    ]
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )

def validate_output_table(
    dataframe,
    table_name,
):
    if len(dataframe) != EXPECTED_TABLE_ROWS:
        raise ValueError(
            f"{table_name}: expected "
            f"{EXPECTED_TABLE_ROWS} rows, "
            f"found {len(dataframe)}."
        )

    if dataframe[
        [
            "outcome",
            "model",
        ]
    ].duplicated().any():
        raise ValueError(
            f"{table_name}: duplicate rows."
        )

    expected_columns = {
        "outcome",
        "model",
        *COMPARISON_LEVELS,
    }

    if set(
        dataframe.columns
    ) != expected_columns:
        raise ValueError(
            f"{table_name}: unexpected columns."
        )

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

def main():
    dataframe = load_data()

    validate_data(
        dataframe
    )

    results_table = (
        build_results_table(
            dataframe
        )
    )

    support_table = (
        build_support_table(
            dataframe
        )
    )

    validate_output_table(
        results_table,
        "primary_results",
    )

    validate_output_table(
        support_table,
        "primary_results_support",
    )

    results_filename = write_table(
        results_table,
        "primary_results_table.csv",
    )

    support_filename = write_table(
        support_table,
        "primary_results_support.csv",
    )

    print(
        "Primary results table "
        "prepared and validated successfully."
    )

    print(
        f"results_rows: "
        f"{len(results_table)}"
    )

    print(
        f"support_rows: "
        f"{len(support_table)}"
    )

    print()

    print(
        "Primary results:"
    )

    print(
        results_table.to_string(
            index=False
        )
    )

    print()

    print(
        "Support:"
    )

    print(
        support_table.to_string(
            index=False
        )
    )

    print()

    print(
        f"saved: {results_filename}"
    )

    print(
        f"saved: {support_filename}"
    )


if __name__ == "__main__":
    main()



