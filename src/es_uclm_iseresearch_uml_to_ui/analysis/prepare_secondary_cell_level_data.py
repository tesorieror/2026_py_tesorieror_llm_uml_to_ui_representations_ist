from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILENAME = Path(
    "experiment/analysis/data/secondary/"
    "association_cardinality_evaluability.csv"
)

OUTPUT_PATH = Path(
    "experiment/analysis/data/secondary/cell_level"
)

OUTPUT_FILENAME = (
    OUTPUT_PATH
    / "association_cardinality_evaluability.csv"
)


EXPECTED_ATTEMPT_ROWS = 1008
EXPECTED_CELLS = 108

EXPECTED_CONSTANT_NUMERATOR_CELLS = 94
EXPECTED_VARIABLE_NUMERATOR_CELLS = 14
EXPECTED_VARIABLE_DENOMINATOR_CELLS = 12


REQUIRED_COLUMNS = [
    "model",
    "uml_model",
    "representation",
    "iteration",
    "cell_id",
    "metric",
    "numerator",
    "denominator",
    "n_not_success",
]

def load_data():
    if not INPUT_FILENAME.is_file():
        raise FileNotFoundError(
            f"Secondary analytical dataset "
            f"not found: {INPUT_FILENAME}"
        )

    return pd.read_csv(
        INPUT_FILENAME
    )

def validate_attempt_level_data(
    dataframe,
):
    missing_columns = (
        set(REQUIRED_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing secondary analytical "
            "columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != EXPECTED_ATTEMPT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ATTEMPT_ROWS} "
            f"attempt rows, found "
            f"{len(dataframe)}."
        )

    if dataframe[
        [
            "model",
            "uml_model",
            "representation",
            "iteration",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate experimental attempts."
        )

    if dataframe[
        [
            "model",
            "uml_model",
            "representation",
            "iteration",
            "cell_id",
            "metric",
            "numerator",
            "denominator",
            "n_not_success",
        ]
    ].isna().any().any():
        raise ValueError(
            "Unexpected NA in secondary "
            "analytical dataset."
        )

    if (
        dataframe["denominator"] <= 0
    ).any():
        raise ValueError(
            "Denominator must be positive."
        )

    if (
        dataframe["numerator"] < 0
    ).any():
        raise ValueError(
            "Numerator cannot be negative."
        )

    if (
        dataframe["numerator"]
        > dataframe["denominator"]
    ).any():
        raise ValueError(
            "Numerator exceeds denominator."
        )

    expected_not_success = (
        dataframe["denominator"]
        - dataframe["numerator"]
    )

    if not np.array_equal(
        dataframe[
            "n_not_success"
        ].to_numpy(),
        expected_not_success.to_numpy(),
    ):
        raise ValueError(
            "n_not_success does not equal "
            "denominator - numerator."
        )

    reconstructed_metric = (
        dataframe["numerator"]
        / dataframe["denominator"]
    )

    if not np.allclose(
        dataframe["metric"].to_numpy(
            dtype=float
        ),
        reconstructed_metric.to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Metric does not match "
            "numerator / denominator."
        )

    if (
        dataframe["cell_id"].nunique()
        != EXPECTED_CELLS
    ):
        raise ValueError(
            f"Expected {EXPECTED_CELLS} cells, "
            f"found "
            f"{dataframe['cell_id'].nunique()}."
        )

    cell_identity = (
        dataframe
        .groupby("cell_id")[
            [
                "model",
                "uml_model",
                "representation",
            ]
        ]
        .nunique()
    )

    if not (
        cell_identity == 1
    ).all().all():
        raise ValueError(
            "cell_id does not uniquely identify "
            "model × UML × representation."
        )

    cell_sizes = (
        dataframe
        .groupby("cell_id")
        .size()
    )

    if (
        cell_sizes < 1
    ).any():
        raise ValueError(
            "Invalid empty analytical cell."
        )

    if (
        cell_sizes > 10
    ).any():
        raise ValueError(
            "A cell contains more than "
            "10 attempts."
        )

def build_cell_level_data(
    dataframe,
):
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
                "iteration",
                "size",
            ),
            numerator_sum=(
                "numerator",
                "sum",
            ),
            denominator_sum=(
                "denominator",
                "sum",
            ),
            numerator_mean=(
                "numerator",
                "mean",
            ),
            denominator_mean=(
                "denominator",
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
            denominator_min=(
                "denominator",
                "min",
            ),
            denominator_max=(
                "denominator",
                "max",
            ),
            numerator_nunique=(
                "numerator",
                "nunique",
            ),
            denominator_nunique=(
                "denominator",
                "nunique",
            ),
        )
    )

    cell_data["cell_ratio"] = (
        cell_data["numerator_sum"]
        / cell_data["denominator_sum"]
    )

    cell_data["numerator_range"] = (
        cell_data["numerator_max"]
        - cell_data["numerator_min"]
    )

    cell_data["denominator_range"] = (
        cell_data["denominator_max"]
        - cell_data["denominator_min"]
    )

    return cell_data


def validate_cell_level_data(
    attempt_data,
    cell_data,
):
    if len(cell_data) != EXPECTED_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_CELLS} "
            f"cell-level rows, found "
            f"{len(cell_data)}."
        )

    if cell_data[
        [
            "cell_id",
            "model",
            "uml_model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate cell-level observations."
        )

    if (
        cell_data["n_attempts"] < 1
    ).any():
        raise ValueError(
            "Invalid n_attempts."
        )

    if (
        cell_data["n_attempts"] > 10
    ).any():
        raise ValueError(
            "Cell contains more than "
            "10 attempts."
        )

    if (
        cell_data["denominator_sum"] <= 0
    ).any():
        raise ValueError(
            "Non-positive aggregated "
            "denominator."
        )

    if (
        cell_data["numerator_sum"] < 0
    ).any():
        raise ValueError(
            "Negative aggregated numerator."
        )

    if (
        cell_data["numerator_sum"]
        > cell_data["denominator_sum"]
    ).any():
        raise ValueError(
            "Aggregated numerator exceeds "
            "aggregated denominator."
        )

    reconstructed_ratio = (
        cell_data["numerator_sum"]
        / cell_data["denominator_sum"]
    )

    if not np.allclose(
        cell_data["cell_ratio"],
        reconstructed_ratio,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Invalid cell_ratio."
        )

    reconstructed_from_means = (
        cell_data["numerator_mean"]
        / cell_data["denominator_mean"]
    )

    if not np.allclose(
        cell_data["cell_ratio"],
        reconstructed_from_means,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "cell_ratio does not match "
            "numerator_mean / denominator_mean."
        )

    if (
        cell_data["numerator_sum"].sum()
        != attempt_data["numerator"].sum()
    ):
        raise ValueError(
            "Cell aggregation does not "
            "preserve numerator total."
        )

    if (
        cell_data["denominator_sum"].sum()
        != attempt_data["denominator"].sum()
    ):
        raise ValueError(
            "Cell aggregation does not "
            "preserve denominator total."
        )

    attempt_element_weighted = (
        attempt_data["numerator"].sum()
        / attempt_data["denominator"].sum()
    )

    cell_element_weighted = (
        cell_data["numerator_sum"].sum()
        / cell_data["denominator_sum"].sum()
    )

    if not np.isclose(
        attempt_element_weighted,
        cell_element_weighted,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Element-weighted estimand "
            "was not preserved."
        )

    constant_numerator_cells = int(
        (
            cell_data[
                "numerator_nunique"
            ] == 1
        ).sum()
    )

    variable_numerator_cells = int(
        (
            cell_data[
                "numerator_nunique"
            ] > 1
        ).sum()
    )

    variable_denominator_cells = int(
        (
            cell_data[
                "denominator_nunique"
            ] > 1
        ).sum()
    )

    if (
        constant_numerator_cells
        != EXPECTED_CONSTANT_NUMERATOR_CELLS
    ):
        raise ValueError(
            "Unexpected number of "
            "constant-numerator cells."
        )

    if (
        variable_numerator_cells
        != EXPECTED_VARIABLE_NUMERATOR_CELLS
    ):
        raise ValueError(
            "Unexpected number of "
            "variable-numerator cells."
        )

    if (
        variable_denominator_cells
        != EXPECTED_VARIABLE_DENOMINATOR_CELLS
    ):
        raise ValueError(
            "Unexpected number of "
            "variable-denominator cells."
        )

    if (
        cell_data["numerator_nunique"]
        > 2
    ).any():
        raise ValueError(
            "A cell contains more than "
            "two distinct numerator values."
        )

def write_cell_level_data(
    cell_data,
):
    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    cell_data.to_csv(
        OUTPUT_FILENAME,
        index=False,
    )

    return OUTPUT_FILENAME

def main():
    attempt_data = load_data()

    validate_attempt_level_data(
        attempt_data
    )

    cell_data = build_cell_level_data(
        attempt_data
    )

    validate_cell_level_data(
        attempt_data,
        cell_data,
    )

    filename = write_cell_level_data(
        cell_data
    )

    element_weighted = (
        cell_data["numerator_sum"].sum()
        / cell_data["denominator_sum"].sum()
    )

    print(
        "Secondary cell-level dataset "
        "prepared and validated successfully."
    )

    print(
        f"attempt_rows: "
        f"{len(attempt_data)}"
    )

    print(
        f"cells: "
        f"{len(cell_data)}"
    )

    print(
        "constant_numerator_cells: "
        f"{int((cell_data['numerator_nunique'] == 1).sum())}"
    )

    print(
        "variable_numerator_cells: "
        f"{int((cell_data['numerator_nunique'] > 1).sum())}"
    )

    print(
        "variable_denominator_cells: "
        f"{int((cell_data['denominator_nunique'] > 1).sum())}"
    )

    print(
        "element_weighted: "
        f"{element_weighted:.12f}"
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

    print(
        f"saved: {filename}"
    )


if __name__ == "__main__":
    main()