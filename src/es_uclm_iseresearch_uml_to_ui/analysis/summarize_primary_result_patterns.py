from pathlib import Path

import numpy as np
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

ZERO_TOLERANCE = 1e-15

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
            "Missing columns: "
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
            "Duplicate contrast rows."
        )

def classify_contrast(
    row,
):
    status = row[
        "analysis_status"
    ]

    if status == "not_estimable":
        return "not_estimable"

    if status == "descriptive_only":
        return "descriptive_only"

    if status != "bayesian_bootstrap":
        raise ValueError(
            f"Unexpected analysis status: "
            f"{status}"
        )

    probability_zero = row[
        "element_bb_probability_eq_0"
    ]

    if np.isclose(
        probability_zero,
        1.0,
        rtol=0.0,
        atol=ZERO_TOLERANCE,
    ):
        return "exact_zero"

    lower = row[
        "element_bb_q025"
    ]

    upper = row[
        "element_bb_q975"
    ]

    if lower > 0:
        return "above_zero"

    if upper < 0:
        return "below_zero"

    return "contains_zero"

def add_pattern_classification(
    dataframe,
):
    result = dataframe.copy()

    result[
        "result_pattern"
    ] = result.apply(
        classify_contrast,
        axis=1,
    )

    return result

def summarize_by_outcome(
    dataframe,
):
    result = (
        dataframe
        .groupby(
            [
                "outcome",
                "result_pattern",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            OUTCOME_LEVELS
        )
        .reset_index()
    )

    return result

def summarize_by_model(
    dataframe,
):
    result = (
        dataframe
        .groupby(
            [
                "model",
                "result_pattern",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            MODEL_LEVELS
        )
        .reset_index()
    )

    return result

def summarize_by_representation(
    dataframe,
):
    result = (
        dataframe
        .groupby(
            [
                "comparison",
                "result_pattern",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            COMPARISON_LEVELS
        )
        .reset_index()
    )

    return result


def validate_classification(
    dataframe,
):
    allowed_patterns = {
        "above_zero",
        "below_zero",
        "exact_zero",
        "contains_zero",
        "descriptive_only",
        "not_estimable",
    }

    if not set(
        dataframe["result_pattern"]
    ).issubset(
        allowed_patterns
    ):
        raise ValueError(
            "Unexpected result pattern."
        )

    exact_zero = dataframe[
        dataframe["result_pattern"]
        == "exact_zero"
    ]

    if not (
        exact_zero[
            "element_bb_probability_eq_0"
        ] == 1.0
    ).all():
        raise ValueError(
            "Invalid exact-zero classification."
        )

    above = dataframe[
        dataframe["result_pattern"]
        == "above_zero"
    ]

    if not (
        above[
            "element_bb_q025"
        ] > 0
    ).all():
        raise ValueError(
            "Invalid above-zero classification."
        )

    below = dataframe[
        dataframe["result_pattern"]
        == "below_zero"
    ]

    if not (
        below[
            "element_bb_q975"
        ] < 0
    ).all():
        raise ValueError(
            "Invalid below-zero classification."
        )

    if (
        dataframe[
            "result_pattern"
        ]
        .value_counts()
        .sum()
        != EXPECTED_ROWS
    ):
        raise ValueError(
            "Classification does not cover "
            "all contrasts."
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

    classified = (
        add_pattern_classification(
            dataframe
        )
    )

    validate_classification(
        classified
    )

    by_outcome = summarize_by_outcome(
        classified
    )

    by_model = summarize_by_model(
        classified
    )

    by_representation = (
        summarize_by_representation(
            classified
        )
    )

    classified_filename = write_table(
        classified,
        "primary_contrast_patterns.csv",
    )

    outcome_filename = write_table(
        by_outcome,
        "primary_contrast_patterns_by_outcome.csv",
    )

    model_filename = write_table(
        by_model,
        "primary_contrast_patterns_by_model.csv",
    )

    representation_filename = write_table(
        by_representation,
        "primary_contrast_patterns_by_representation.csv",
    )

    print(
        "Primary contrast pattern summary "
        "completed and validated successfully."
    )

    print()
    print(
        "Overall patterns:"
    )

    print(
        classified[
            "result_pattern"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Patterns by outcome:"
    )

    print(
        by_outcome.to_string(
            index=False
        )
    )

    print()
    print(
        "Patterns by model:"
    )

    print(
        by_model.to_string(
            index=False
        )
    )

    print()
    print(
        "Patterns by representation:"
    )

    print(
        by_representation.to_string(
            index=False
        )
    )

    print()
    print(
        f"saved: {classified_filename}"
    )

    print(
        f"saved: {outcome_filename}"
    )

    print(
        f"saved: {model_filename}"
    )

    print(
        f"saved: {representation_filename}"
    )


if __name__ == "__main__":
    main()