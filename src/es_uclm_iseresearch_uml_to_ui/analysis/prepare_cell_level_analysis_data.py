from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_DATA_PATH = Path(
    "experiment/analysis/data"
)


OUTCOMES = {
    "attribute_coverage": {
        "expected_attempt_rows": 1479,
        "expected_cells": 148,
        "denominator_constant_within_cell": True,
    },
    "primitive_type_fidelity": {
        "expected_attempt_rows": 1389,
        "expected_cells": 141,
        "denominator_constant_within_cell": False,
    },
    "enumeration_literal_coverage": {
        "expected_attempt_rows": 1040,
        "expected_cells": 104,
        "denominator_constant_within_cell": True,
    },
    "enumeration_cardinality_fidelity": {
        "expected_attempt_rows": 803,
        "expected_cells": 83,
        "denominator_constant_within_cell": False,
    },
    "attribute_nonempty_label_coverage": {
        "expected_attempt_rows": 1389,
        "expected_cells": 141,
        "denominator_constant_within_cell": False,
    },
    "association_end_coverage": {
        "expected_attempt_rows": 1479,
        "expected_cells": 148,
        "denominator_constant_within_cell": True,
    },
    "association_cardinality_fidelity": {
        "expected_attempt_rows": 971,
        "expected_cells": 105,
        "denominator_constant_within_cell": False,
    },
}


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

def load_outcome_data(
    outcome_name,
):
    if outcome_name not in OUTCOMES:
        raise KeyError(
            f"Unknown outcome: {outcome_name}"
        )

    filename = (
        ANALYSIS_DATA_PATH
        / f"{outcome_name}.csv"
    )

    if not filename.is_file():
        raise FileNotFoundError(
            f"Analytical dataset not found: {filename}"
        )

    return pd.read_csv(filename)

def validate_attempt_level_data(
    outcome_name,
    dataframe,
):
    config = OUTCOMES[outcome_name]

    missing_columns = (
        set(REQUIRED_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{outcome_name}: missing columns: "
            f"{sorted(missing_columns)}"
        )

    expected_rows = config[
        "expected_attempt_rows"
    ]

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"{outcome_name}: expected "
            f"{expected_rows} attempt-level rows, "
            f"found {len(dataframe)}."
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
            f"{outcome_name}: duplicate "
            "experimental attempts."
        )

    if dataframe[
        [
            "model",
            "uml_model",
            "representation",
            "cell_id",
            "numerator",
            "denominator",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{outcome_name}: unexpected NA "
            "in required analytical values."
        )

    if (
        dataframe["denominator"] <= 0
    ).any():
        raise ValueError(
            f"{outcome_name}: denominator "
            "must be strictly positive."
        )

    if (
        dataframe["numerator"] < 0
    ).any():
        raise ValueError(
            f"{outcome_name}: numerator "
            "cannot be negative."
        )

    if (
        dataframe["numerator"]
        > dataframe["denominator"]
    ).any():
        raise ValueError(
            f"{outcome_name}: numerator "
            "exceeds denominator."
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
            f"{outcome_name}: n_not_success "
            "does not equal denominator - numerator."
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
            f"{outcome_name}: metric does not "
            "match numerator / denominator."
        )

    observed_cells = (
        dataframe["cell_id"].nunique()
    )

    expected_cells = config[
        "expected_cells"
    ]

    if observed_cells != expected_cells:
        raise ValueError(
            f"{outcome_name}: expected "
            f"{expected_cells} evaluable cells, "
            f"found {observed_cells}."
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
            f"{outcome_name}: cell_id does not "
            "uniquely identify "
            "model x UML x representation."
        )

    denominator_nunique = (
        dataframe
        .groupby("cell_id")[
            "denominator"
        ]
        .nunique()
    )

    if config[
        "denominator_constant_within_cell"
    ]:
        if not (
            denominator_nunique == 1
        ).all():
            raise ValueError(
                f"{outcome_name}: denominator "
                "must be constant within cell."
            )

# 
def build_cell_level_data(
    outcome_name,
    dataframe,
):
    config = OUTCOMES[outcome_name]

    grouped = (
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

    grouped["cell_ratio"] = (
        grouped["numerator_sum"]
        / grouped["denominator_sum"]
    )

    grouped["numerator_range"] = (
        grouped["numerator_max"]
        - grouped["numerator_min"]
    )

    grouped["denominator_range"] = (
        grouped["denominator_max"]
        - grouped["denominator_min"]
    )

    if config[
        "denominator_constant_within_cell"
    ]:
        grouped["denominator_per_attempt"] = (
            grouped["denominator_min"]
        )
    else:
        grouped["denominator_per_attempt"] = np.nan

    return grouped

def validate_cell_level_data(
    outcome_name,
    attempt_data,
    cell_data,
):
    config = OUTCOMES[outcome_name]

    expected_cells = config[
        "expected_cells"
    ]

    if len(cell_data) != expected_cells:
        raise ValueError(
            f"{outcome_name}: expected "
            f"{expected_cells} cell-level rows, "
            f"found {len(cell_data)}."
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
            f"{outcome_name}: duplicate "
            "cell-level rows."
        )

    if (
        cell_data["n_attempts"] <= 0
    ).any():
        raise ValueError(
            f"{outcome_name}: invalid "
            "number of attempts."
        )

    if (
        cell_data["denominator_sum"] <= 0
    ).any():
        raise ValueError(
            f"{outcome_name}: non-positive "
            "cell denominator."
        )

    if (
        cell_data["numerator_sum"] < 0
    ).any():
        raise ValueError(
            f"{outcome_name}: negative "
            "cell numerator."
        )

    if (
        cell_data["numerator_sum"]
        > cell_data["denominator_sum"]
    ).any():
        raise ValueError(
            f"{outcome_name}: cell numerator "
            "exceeds cell denominator."
        )

    if not np.allclose(
        cell_data["cell_ratio"],
        (
            cell_data["numerator_sum"]
            / cell_data["denominator_sum"]
        ),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"{outcome_name}: invalid "
            "cell_ratio."
        )

    if (
        cell_data["numerator_sum"].sum()
        != attempt_data["numerator"].sum()
    ):
        raise ValueError(
            f"{outcome_name}: aggregation "
            "does not preserve numerator total."
        )

    if (
        cell_data["denominator_sum"].sum()
        != attempt_data["denominator"].sum()
    ):
        raise ValueError(
            f"{outcome_name}: aggregation "
            "does not preserve denominator total."
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
            f"{outcome_name}: element-weighted "
            "estimand not preserved."
        )

    if config[
        "denominator_constant_within_cell"
    ]:
        if not (
            cell_data["denominator_nunique"]
            == 1
        ).all():
            raise ValueError(
                f"{outcome_name}: denominator "
                "must be constant within cell."
            )

        expected_denominator_sum = (
            cell_data["n_attempts"]
            * cell_data["denominator_per_attempt"]
        )

        if not np.array_equal(
            cell_data[
                "denominator_sum"
            ].to_numpy(),
            expected_denominator_sum.to_numpy(),
        ):
            raise ValueError(
                f"{outcome_name}: invalid "
                "denominator aggregation."
            )

CELL_LEVEL_DATA_PATH = (
    ANALYSIS_DATA_PATH
    / "cell_level"
)

def write_cell_level_data(
    outcome_name,
    cell_data,
):
    CELL_LEVEL_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        CELL_LEVEL_DATA_PATH
        / f"{outcome_name}.csv"
    )

    cell_data.to_csv(
        filename,
        index=False,
    )

    return filename

def main():
    for outcome_name in OUTCOMES:
        attempt_data = load_outcome_data(
            outcome_name
        )

        validate_attempt_level_data(
            outcome_name,
            attempt_data,
        )

        cell_data = build_cell_level_data(
            outcome_name,
            attempt_data,
        )

        validate_cell_level_data(
            outcome_name,
            attempt_data,
            cell_data,
        )

        filename = write_cell_level_data(
            outcome_name,
            cell_data,
        )

        variable_denominator_cells = int(
            (
                cell_data[
                    "denominator_nunique"
                ]
                > 1
            ).sum()
        )

        constant_numerator_cells = int(
            (
                cell_data[
                    "numerator_nunique"
                ]
                == 1
            ).sum()
        )

        print(
            f"{outcome_name}: "
            f"{len(attempt_data)} attempts -> "
            f"{len(cell_data)} cells; "
            f"{constant_numerator_cells} "
            "constant-numerator cells; "
            f"{variable_denominator_cells} "
            "variable-denominator cells"
        )

        print(
            "element_weighted: "
            f"{cell_data['numerator_sum'].sum() / cell_data['denominator_sum'].sum():.12f}"
        )

        print(
            f"saved: {filename}"
        )

    print(
        "Cell-level analytical datasets "
        "prepared and validated successfully."
    )


if __name__ == "__main__":
    main()


