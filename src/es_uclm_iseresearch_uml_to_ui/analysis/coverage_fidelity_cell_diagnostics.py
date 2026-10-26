from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_DATA_PATH = Path(
    "experiment/analysis/data"
)


OUTCOMES = [
    "attribute_coverage",
    "primitive_type_fidelity",
    "enumeration_literal_coverage",
    "enumeration_cardinality_fidelity",
    "attribute_nonempty_label_coverage",
    "association_end_coverage",
    "association_cardinality_fidelity",
]


REQUIRED_COLUMNS = {
    "model",
    "uml_model",
    "representation",
    "iteration",
    "cell_id",
    "numerator",
    "denominator",
}


def load_outcome_data(
    outcome_name,
):
    filename = (
        ANALYSIS_DATA_PATH
        / f"{outcome_name}.csv"
    )

    if not filename.is_file():
        raise FileNotFoundError(
            f"Outcome dataset not found: "
            f"{filename}"
        )

    dataframe = pd.read_csv(
        filename
    )

    return dataframe


def validate_outcome_data(
    outcome_name,
    dataframe,
):
    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{outcome_name}: missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) == 0:
        raise ValueError(
            f"{outcome_name}: empty dataset."
        )

    duplicated = dataframe.duplicated(
        subset=[
            "model",
            "uml_model",
            "representation",
            "iteration",
        ],
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            f"{outcome_name}: duplicate "
            "experimental attempts."
        )

    if dataframe[
        "numerator"
    ].isna().any():
        raise ValueError(
            f"{outcome_name}: numerator "
            "contains NA."
        )

    if dataframe[
        "denominator"
    ].isna().any():
        raise ValueError(
            f"{outcome_name}: denominator "
            "contains NA."
        )

    if (
        dataframe["denominator"] <= 0
    ).any():
        raise ValueError(
            f"{outcome_name}: non-positive "
            "denominator."
        )

    if (
        dataframe["numerator"] < 0
    ).any():
        raise ValueError(
            f"{outcome_name}: negative "
            "numerator."
        )

    if (
        dataframe["numerator"]
        > dataframe["denominator"]
    ).any():
        raise ValueError(
            f"{outcome_name}: numerator "
            "exceeds denominator."
        )


def build_cell_diagnostics(
    outcome_name,
    dataframe,
):
    grouped = dataframe.groupby(
        [
            "cell_id",
            "model",
            "uml_model",
            "representation",
        ],
        as_index=False,
    )

    cell_data = grouped.agg(
        n_attempts=(
            "numerator",
            "size",
        ),
        numerator_sum=(
            "numerator",
            "sum",
        ),
        numerator_mean=(
            "numerator",
            "mean",
        ),
        numerator_min=(
            "numerator",
            "min",
        ),
        numerator_max=(
            "numerator",
            "max",
        ),
        numerator_variance=(
            "numerator",
            "var",
        ),
        numerator_nunique=(
            "numerator",
            "nunique",
        ),
        denominator_sum=(
            "denominator",
            "sum",
        ),
        denominator_mean=(
            "denominator",
            "mean",
        ),
        denominator_min=(
            "denominator",
            "min",
        ),
        denominator_max=(
            "denominator",
            "max",
        ),
        denominator_nunique=(
            "denominator",
            "nunique",
        ),
    )

    cell_data["numerator_range"] = (
        cell_data["numerator_max"]
        - cell_data["numerator_min"]
    )

    cell_data["cell_ratio"] = (
        cell_data["numerator_sum"]
        / cell_data["denominator_sum"]
    )

    if not (
        cell_data["cell_ratio"]
        .between(
            0.0,
            1.0,
            inclusive="both",
        )
    ).all():
        raise ValueError(
            f"{outcome_name}: invalid "
            "cell-level ratio."
        )

    return cell_data


def validate_cell_aggregation(
    outcome_name,
    dataframe,
    cell_data,
):
    if not np.isclose(
        cell_data[
            "numerator_sum"
        ].sum(),
        dataframe[
            "numerator"
        ].sum(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{outcome_name}: aggregation "
            "does not preserve numerator."
        )

    if not np.isclose(
        cell_data[
            "denominator_sum"
        ].sum(),
        dataframe[
            "denominator"
        ].sum(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{outcome_name}: aggregation "
            "does not preserve denominator."
        )

    attempt_element_weighted = (
        dataframe["numerator"].sum()
        / dataframe["denominator"].sum()
    )

    cell_aggregated_element_weighted = (
        cell_data["numerator_sum"].sum()
        / cell_data["denominator_sum"].sum()
    )

    if not np.isclose(
        attempt_element_weighted,
        cell_aggregated_element_weighted,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"{outcome_name}: element-weighted "
            "estimate changed after aggregation."
        )


def summarize_outcome(
    outcome_name,
    dataframe,
    cell_data,
):
    constant_numerator_cells = int(
        (
            cell_data[
                "numerator_nunique"
            ] == 1
        ).sum()
    )

    two_value_numerator_cells = int(
        (
            cell_data[
                "numerator_nunique"
            ] == 2
        ).sum()
    )

    more_than_two_value_cells = int(
        (
            cell_data[
                "numerator_nunique"
            ] > 2
        ).sum()
    )

    variable_denominator_cells = int(
        (
            cell_data[
                "denominator_nunique"
            ] > 1
        ).sum()
    )

    zero_coverage_cells = int(
        (
            cell_data["cell_ratio"]
            == 0
        ).sum()
    )

    perfect_coverage_cells = int(
        (
            cell_data["cell_ratio"]
            == 1
        ).sum()
    )

    maximum_numerator_range = float(
        cell_data[
            "numerator_range"
        ].max()
    )

    element_weighted = float(
        dataframe["numerator"].sum()
        / dataframe["denominator"].sum()
    )

    return {
        "outcome": outcome_name,
        "attempts": len(dataframe),
        "cells": len(cell_data),
        "constant_numerator_cells": (
            constant_numerator_cells
        ),
        "two_value_numerator_cells": (
            two_value_numerator_cells
        ),
        "more_than_two_value_cells": (
            more_than_two_value_cells
        ),
        "variable_denominator_cells": (
            variable_denominator_cells
        ),
        "zero_ratio_cells": (
            zero_coverage_cells
        ),
        "perfect_ratio_cells": (
            perfect_coverage_cells
        ),
        "maximum_numerator_range": (
            maximum_numerator_range
        ),
        "element_weighted_ratio": (
            element_weighted
        ),
    }


def print_detailed_diagnostics(
    outcome_name,
    cell_data,
):
    print()
    print("=" * 80)
    print(outcome_name)
    print("=" * 80)

    print(
        "n_attempts_distribution:"
    )

    print(
        cell_data[
            "n_attempts"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "numerator_nunique_distribution:"
    )

    print(
        cell_data[
            "numerator_nunique"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "denominator_nunique_distribution:"
    )

    print(
        cell_data[
            "denominator_nunique"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    variable_cells = cell_data[
        cell_data[
            "numerator_nunique"
        ] > 1
    ].copy()

    if len(variable_cells) > 0:
        print()
        print(
            "variable_cell_range_distribution:"
        )

        print(
            variable_cells[
                "numerator_range"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print()
        print(
            "maximum_variable_cell_range: "
            f"{variable_cells['numerator_range'].max()}"
        )


def main():
    summaries = []

    for outcome_name in OUTCOMES:
        dataframe = load_outcome_data(
            outcome_name
        )

        validate_outcome_data(
            outcome_name,
            dataframe,
        )

        cell_data = (
            build_cell_diagnostics(
                outcome_name,
                dataframe,
            )
        )

        validate_cell_aggregation(
            outcome_name,
            dataframe,
            cell_data,
        )

        summary = summarize_outcome(
            outcome_name,
            dataframe,
            cell_data,
        )

        summaries.append(
            summary
        )

        print_detailed_diagnostics(
            outcome_name,
            cell_data,
        )

    summary_dataframe = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 80)
    print(
        "COVERAGE/FIDELITY CELL DIAGNOSTIC SUMMARY"
    )
    print("=" * 80)

    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    print()
    print(
        "Coverage/fidelity cell diagnostics "
        "completed successfully."
    )


if __name__ == "__main__":
    main()