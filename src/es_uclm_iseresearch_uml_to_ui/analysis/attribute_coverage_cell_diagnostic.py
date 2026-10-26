from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILENAME = Path(
    "experiment/analysis/data/"
    "attribute_coverage.csv"
)


def load_data():
    if not DATA_FILENAME.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILENAME}"
        )

    return pd.read_csv(DATA_FILENAME)


def build_cell_data(dataframe):
    denominator_variation = (
        dataframe
        .groupby("cell_id")["denominator"]
        .nunique()
    )

    if not (
        denominator_variation == 1
    ).all():
        raise ValueError(
            "Denominator must be constant "
            "within every cell."
        )

    cell_data = (
        dataframe
        .groupby(
            [
                "cell_id",
                "model",
                "uml_model",
                "representation",
            ],
            as_index=False,
        )
        .agg(
            n_attempts=(
                "numerator",
                "size",
            ),
            denominator_per_attempt=(
                "denominator",
                "first",
            ),
            numerator_sum=(
                "numerator",
                "sum",
            ),
            numerator_mean=(
                "numerator",
                "mean",
            ),
            numerator_variance=(
                "numerator",
                "var",
            ),
            numerator_nunique=(
                "numerator",
                "nunique",
            ),
        )
    )

    cell_data["denominator_sum"] = (
        cell_data["n_attempts"]
        * cell_data[
            "denominator_per_attempt"
        ]
    )

    cell_data["cell_coverage"] = (
        cell_data["numerator_sum"]
        / cell_data["denominator_sum"]
    )

    return cell_data


def validate_cell_data(
    dataframe,
    cell_data,
):
    if len(cell_data) != 148:
        raise ValueError(
            "Expected 148 evaluable cells, "
            f"found {len(cell_data)}."
        )

    if not cell_data[
        "n_attempts"
    ].isin([9, 10]).all():
        raise ValueError(
            "Every evaluable cell must contain "
            "9 or 10 attempts."
        )

    if (
        cell_data["numerator_sum"].sum()
        != dataframe["numerator"].sum()
    ):
        raise ValueError(
            "Cell aggregation does not preserve "
            "the total numerator."
        )

    if (
        cell_data["denominator_sum"].sum()
        != dataframe["denominator"].sum()
    ):
        raise ValueError(
            "Cell aggregation does not preserve "
            "the total denominator."
        )

    reconstructed = (
        cell_data["numerator_sum"]
        / cell_data["denominator_sum"]
    )

    if not np.allclose(
        reconstructed,
        cell_data["cell_coverage"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Invalid cell coverage."
        )

    variable_cells = (
        dataframe
        .groupby(
            [
                "cell_id",
                "model",
                "uml_model",
                "representation",
            ]
        )
        .agg(
            denominator=(
                "denominator",
                "first",
            ),
            minimum=(
                "numerator",
                "min",
            ),
            maximum=(
                "numerator",
                "max",
            ),
            mean=(
                "numerator",
                "mean",
            ),
            standard_deviation=(
                "numerator",
                "std",
            ),
            nunique=(
                "numerator",
                "nunique",
            ),
        )
        .reset_index()
    )

    variable_cells = (
        variable_cells[
            variable_cells["nunique"] > 1
        ]
        .copy()
    )

    variable_cells["range"] = (
        variable_cells["maximum"]
        - variable_cells["minimum"]
    )

    print(
        "variable_cell_range_distribution:"
    )

    print(
        variable_cells["range"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "maximum_variable_cell_range: "
        f"{variable_cells['range'].max()}"
    )

    print(
        "variable_cells:"
    )

    print(
        variable_cells[
            [
                "model",
                "uml_model",
                "representation",
                "denominator",
                "minimum",
                "maximum",
                "mean",
                "standard_deviation",
                "range",
            ]
        ].to_string(index=False)
    )

def main():
    dataframe = load_data()

    cell_data = build_cell_data(
        dataframe
    )

    validate_cell_data(
        dataframe,
        cell_data,
    )

    n_constant = int(
        (
            cell_data["numerator_nunique"]
            == 1
        ).sum()
    )

    n_two_values = int(
        (
            cell_data["numerator_nunique"]
            == 2
        ).sum()
    )

    n_zero_cells = int(
        (
            cell_data["cell_coverage"]
            == 0
        ).sum()
    )

    n_perfect_cells = int(
        (
            cell_data["cell_coverage"]
            == 1
        ).sum()
    )

    observed_element_weighted = (
        dataframe["numerator"].sum()
        / dataframe["denominator"].sum()
    )

    aggregated_element_weighted = (
        cell_data["numerator_sum"].sum()
        / cell_data["denominator_sum"].sum()
    )

    print(
        "Attribute-coverage cell diagnostic "
        "completed successfully."
    )

    print(
        f"attempt_level_rows: "
        f"{len(dataframe)}"
    )

    print(
        f"evaluable_cells: "
        f"{len(cell_data)}"
    )

    print(
        f"constant_numerator_cells: "
        f"{n_constant}"
    )

    print(
        f"two_value_numerator_cells: "
        f"{n_two_values}"
    )

    print(
        f"zero_coverage_cells: "
        f"{n_zero_cells}"
    )

    print(
        f"perfect_coverage_cells: "
        f"{n_perfect_cells}"
    )

    print(
        "attempt_level_element_weighted: "
        f"{observed_element_weighted:.12f}"
    )

    print(
        "cell_aggregated_element_weighted: "
        f"{aggregated_element_weighted:.12f}"
    )

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






if __name__ == "__main__":
    main()