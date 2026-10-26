from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILENAME = Path(
    "experiment/analysis/results/"
    "cell_level_bayesian_bootstrap/"
    "primary_contrasts.csv"
)

OUTPUT_PATH = Path(
    "experiment/analysis/tables"
)


EXPECTED_ROWS = 84


ANALYSIS_STATUS_LEVELS = {
    "bayesian_bootstrap",
    "descriptive_only",
    "not_estimable",
}

def load_contrasts():
    if not INPUT_FILENAME.is_file():
        raise FileNotFoundError(
            f"Primary contrasts not found: "
            f"{INPUT_FILENAME}"
        )

    return pd.read_csv(
        INPUT_FILENAME
    )

def validate_source_data(
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

        "reference_element_weighted",
        "comparison_element_weighted",
        "element_weighted_difference",

        "element_bb_mean",
        "element_bb_median",
        "element_bb_q025",
        "element_bb_q975",
        "element_bb_probability_gt_0",
        "element_bb_probability_lt_0",
        "element_bb_probability_eq_0",

        "reference_model_weighted",
        "comparison_model_weighted",
        "model_weighted_difference",

        "model_weighted_bb_mean",
        "model_weighted_bb_median",
        "model_weighted_bb_q025",
        "model_weighted_bb_q975",
        "model_weighted_bb_probability_gt_0",
        "model_weighted_bb_probability_lt_0",
        "model_weighted_bb_probability_eq_0",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required contrast columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} contrasts, "
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
            "Duplicate contrast rows."
        )

    if not set(
        dataframe["analysis_status"]
    ).issubset(
        ANALYSIS_STATUS_LEVELS
    ):
        raise ValueError(
            "Unexpected analysis_status."
        )

    if not (
        dataframe["reference"] == "uml"
    ).all():
        raise ValueError(
            "Unexpected reference representation."
        )


    element_difference = (
        dataframe[
            "comparison_element_weighted"
        ]
        - dataframe[
            "reference_element_weighted"
        ]
    )

    element_mask = (
        dataframe[
            "element_weighted_difference"
        ].notna()
    )

    if not np.allclose(
        dataframe.loc[
            element_mask,
            "element_weighted_difference",
        ],
        element_difference[
            element_mask
        ],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Invalid element-weighted "
            "descriptive differences."
        )

    model_difference = (
        dataframe[
            "comparison_model_weighted"
        ]
        - dataframe[
            "reference_model_weighted"
        ]
    )

    model_mask = (
        dataframe[
            "model_weighted_difference"
        ].notna()
    )

    if not np.allclose(
        dataframe.loc[
            model_mask,
            "model_weighted_difference",
        ],
        model_difference[
            model_mask
        ],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Invalid model-weighted "
            "descriptive differences."
        )    


    not_estimable = (
        dataframe[
            "analysis_status"
        ] == "not_estimable"
    )

    descriptive_only = (
        dataframe[
            "analysis_status"
        ] == "descriptive_only"
    )

    bootstrap = (
        dataframe[
            "analysis_status"
        ] == "bayesian_bootstrap"
    )

    if not (
        dataframe.loc[
            not_estimable,
            "n_paired_umls",
        ] == 0
    ).all():
        raise ValueError(
            "not_estimable contrasts must "
            "have zero paired UMLs."
        )

    if not (
        dataframe.loc[
            descriptive_only,
            "n_paired_umls",
        ] == 1
    ).all():
        raise ValueError(
            "descriptive_only contrasts must "
            "have exactly one paired UML."
        )

    if not (
        dataframe.loc[
            bootstrap,
            "n_paired_umls",
        ] >= 2
    ).all():
        raise ValueError(
            "Bayesian-bootstrap contrasts must "
            "have at least two paired UMLs."
        )

def build_primary_element_weighted_table(
    dataframe,
):
    columns = [
        "outcome",
        "model",
        "reference",
        "comparison",
        "n_paired_umls",
        "paired_umls",
        "analysis_status",

        "reference_element_weighted",
        "comparison_element_weighted",
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

def build_model_weighted_sensitivity_table(
    dataframe,
):
    columns = [
        "outcome",
        "model",
        "reference",
        "comparison",
        "n_paired_umls",
        "paired_umls",
        "analysis_status",

        "reference_model_weighted",
        "comparison_model_weighted",
        "model_weighted_difference",

        "model_weighted_bb_mean",
        "model_weighted_bb_median",
        "model_weighted_bb_q025",
        "model_weighted_bb_q975",
        "model_weighted_bb_probability_gt_0",
        "model_weighted_bb_probability_lt_0",
        "model_weighted_bb_probability_eq_0",
    ]

    result = dataframe[
        columns
    ].copy()

    return result

def validate_output_table(
    dataframe,
    table_name,
):
    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"{table_name}: expected "
            f"{EXPECTED_ROWS} rows, "
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
            f"{table_name}: duplicate rows."
        )

    if dataframe[
        [
            "outcome",
            "model",
            "comparison",
            "n_paired_umls",
            "analysis_status",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{table_name}: unexpected NA "
            "in structural columns."
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

def interval_position(
    lower,
    upper,
):
    if pd.isna(lower) or pd.isna(upper):
        return "not_available"

    if lower > 0:
        return "above_zero"

    if upper < 0:
        return "below_zero"

    return "crosses_zero"

def summarize_interval_positions(
    dataframe,
    lower_column,
    upper_column,
):
    positions = [
        interval_position(
            lower,
            upper,
        )
        for lower, upper in zip(
            dataframe[lower_column],
            dataframe[upper_column],
        )
    ]

    return (
        pd.Series(
            positions
        )
        .value_counts()
    )

def main():
    contrasts = load_contrasts()

    validate_source_data(
        contrasts
    )

    primary = (
        build_primary_element_weighted_table(
            contrasts
        )
    )

    sensitivity = (
        build_model_weighted_sensitivity_table(
            contrasts
        )
    )

    validate_output_table(
        primary,
        "primary_element_weighted",
    )

    validate_output_table(
        sensitivity,
        "sensitivity_model_weighted",
    )

    primary_filename = write_table(
        primary,
        "primary_element_weighted_contrasts.csv",
    )

    sensitivity_filename = write_table(
        sensitivity,
        "sensitivity_model_weighted_contrasts.csv",
    )

    print(
        "Cell-level contrast tables "
        "prepared and validated successfully."
    )

    print(
        f"primary_rows: {len(primary)}"
    )

    print(
        f"sensitivity_rows: "
        f"{len(sensitivity)}"
    )

    print()

    print(
        "Primary interval positions:"
    )

    print(
        summarize_interval_positions(
            primary,
            "element_bb_q025",
            "element_bb_q975",
        ).to_string()
    )

    print()

    print(
        "Sensitivity interval positions:"
    )

    print(
        summarize_interval_positions(
            sensitivity,
            "model_weighted_bb_q025",
            "model_weighted_bb_q975",
        ).to_string()
    )

    print()

    print(
        f"saved: {primary_filename}"
    )

    print(
        f"saved: {sensitivity_filename}"
    )


if __name__ == "__main__":
    main()