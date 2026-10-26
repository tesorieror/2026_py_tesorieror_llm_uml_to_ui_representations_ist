from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_DATA_PATH = Path(
    "experiment/analysis/data"
)


OUTCOMES = [
    "primitive_type_fidelity",
    "enumeration_cardinality_fidelity",
    "attribute_nonempty_label_coverage",
    "association_cardinality_fidelity",
]


def load_outcome_data(
    outcome_name,
):
    filename = (
        ANALYSIS_DATA_PATH
        / f"{outcome_name}.csv"
    )

    if not filename.is_file():
        raise FileNotFoundError(
            f"Analytical dataset not found: "
            f"{filename}"
        )

    return pd.read_csv(filename)


def validate_outcome_data(
    outcome_name,
    dataframe,
):
    required_columns = {
        "model",
        "uml_model",
        "representation",
        "iteration",
        "cell_id",
        "metric",
        "numerator",
        "denominator",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{outcome_name}: missing columns: "
            f"{sorted(missing_columns)}"
        )

    if (
        dataframe["denominator"] <= 0
    ).any():
        raise ValueError(
            f"{outcome_name}: denominator "
            "must be positive."
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

    reconstructed_metric = (
        dataframe["numerator"]
        / dataframe["denominator"]
    )

    if not np.allclose(
        dataframe["metric"],
        reconstructed_metric,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"{outcome_name}: metric does not "
            "match numerator / denominator."
        )


def build_variable_denominator_diagnostics(
    outcome_name,
    dataframe,
):
    denominator_nunique = (
        dataframe
        .groupby("cell_id")[
            "denominator"
        ]
        .nunique()
    )

    variable_cells = set(
        denominator_nunique[
            denominator_nunique > 1
        ].index
    )

    if not variable_cells:
        return pd.DataFrame(
            columns=[
                "outcome",
                "cell_id",
                "model",
                "uml_model",
                "representation",
                "n_attempts",
                "numerator_sum",
                "denominator_sum",
                "pooled_ratio",
                "attempt_mean_ratio",
                "absolute_difference",
                "signed_difference",
                "denominator_min",
                "denominator_max",
                "denominator_range",
                "denominator_nunique",
            ]
        )

    subset = dataframe[
        dataframe["cell_id"].isin(
            variable_cells
        )
    ].copy()

    grouped = (
        subset
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
            attempt_mean_ratio=(
                "metric",
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
    )

    grouped["pooled_ratio"] = (
        grouped["numerator_sum"]
        / grouped["denominator_sum"]
    )

    grouped["signed_difference"] = (
        grouped["pooled_ratio"]
        - grouped["attempt_mean_ratio"]
    )

    grouped["absolute_difference"] = (
        grouped["signed_difference"].abs()
    )

    grouped["denominator_range"] = (
        grouped["denominator_max"]
        - grouped["denominator_min"]
    )

    grouped.insert(
        0,
        "outcome",
        outcome_name,
    )

    return grouped[
        [
            "outcome",
            "cell_id",
            "model",
            "uml_model",
            "representation",
            "n_attempts",
            "numerator_sum",
            "denominator_sum",
            "pooled_ratio",
            "attempt_mean_ratio",
            "absolute_difference",
            "signed_difference",
            "denominator_min",
            "denominator_max",
            "denominator_range",
            "denominator_nunique",
        ]
    ]


def validate_diagnostics(
    outcome_name,
    dataframe,
    diagnostics,
):
    expected_variable_cells = int(
        (
            dataframe
            .groupby("cell_id")[
                "denominator"
            ]
            .nunique()
            > 1
        ).sum()
    )

    if (
        len(diagnostics)
        != expected_variable_cells
    ):
        raise ValueError(
            f"{outcome_name}: expected "
            f"{expected_variable_cells} "
            "variable-denominator cells, "
            f"found {len(diagnostics)}."
        )

    if diagnostics.empty:
        return

    if not (
        diagnostics["denominator_nunique"]
        > 1
    ).all():
        raise ValueError(
            f"{outcome_name}: diagnostic "
            "contains a constant-denominator cell."
        )

    if not diagnostics[
        "pooled_ratio"
    ].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            f"{outcome_name}: pooled_ratio "
            "outside [0,1]."
        )

    if not diagnostics[
        "attempt_mean_ratio"
    ].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            f"{outcome_name}: "
            "attempt_mean_ratio outside [0,1]."
        )

    reconstructed_absolute = (
        diagnostics[
            "signed_difference"
        ].abs()
    )

    if not np.allclose(
        diagnostics[
            "absolute_difference"
        ],
        reconstructed_absolute,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            f"{outcome_name}: invalid "
            "absolute_difference."
        )


def summarize_diagnostics(
    diagnostics,
):
    if diagnostics.empty:
        return {
            "n_variable_denominator_cells": 0,
            "mean_absolute_difference": 0.0,
            "median_absolute_difference": 0.0,
            "maximum_absolute_difference": 0.0,
        }

    return {
        "n_variable_denominator_cells": int(
            len(diagnostics)
        ),
        "mean_absolute_difference": float(
            diagnostics[
                "absolute_difference"
            ].mean()
        ),
        "median_absolute_difference": float(
            diagnostics[
                "absolute_difference"
            ].median()
        ),
        "maximum_absolute_difference": float(
            diagnostics[
                "absolute_difference"
            ].max()
        ),
    }


def main():
    all_diagnostics = []

    for outcome_name in OUTCOMES:
        dataframe = load_outcome_data(
            outcome_name
        )

        validate_outcome_data(
            outcome_name,
            dataframe,
        )

        diagnostics = (
            build_variable_denominator_diagnostics(
                outcome_name,
                dataframe,
            )
        )

        validate_diagnostics(
            outcome_name,
            dataframe,
            diagnostics,
        )

        summary = summarize_diagnostics(
            diagnostics
        )

        all_diagnostics.append(
            diagnostics
        )

        print()
        print("=" * 80)
        print(outcome_name)
        print("=" * 80)

        for key, value in (
            summary.items()
        ):
            print(
                f"{key}: {value}"
            )

        if not diagnostics.empty:
            print()
            print(
                "Largest absolute differences:"
            )

            print(
                diagnostics
                .sort_values(
                    "absolute_difference",
                    ascending=False,
                )
                .head(10)
                [
                    [
                        "model",
                        "uml_model",
                        "representation",
                        "n_attempts",
                        "pooled_ratio",
                        "attempt_mean_ratio",
                        "absolute_difference",
                        "denominator_min",
                        "denominator_max",
                    ]
                ]
                .to_string(
                    index=False
                )
            )

    combined = pd.concat(
        all_diagnostics,
        ignore_index=True,
    )

    output_filename = (
        ANALYSIS_DATA_PATH
        / "cell_level"
        / "variable_denominator_diagnostics.csv"
    )

    combined.to_csv(
        output_filename,
        index=False,
    )

    print()
    print(
        "Variable-denominator diagnostics "
        "completed successfully."
    )

    print(
        f"rows: {len(combined)}"
    )

    print(
        f"saved: {output_filename}"
    )


if __name__ == "__main__":
    main()