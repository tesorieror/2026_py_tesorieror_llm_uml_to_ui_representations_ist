from pathlib import Path

import numpy as np
import pandas as pd

from es_uclm_iseresearch_uml_to_ui.analysis.prepare_analysis_data import (
    ID_COLUMNS,
    add_cell_id,
    load_canonical_dataset,
    validate_canonical_dataset,
)


SECONDARY_ANALYSIS_DATA_PATH = Path(
    "experiment/analysis/data/secondary"
)


SECONDARY_OUTCOME_NAME = (
    "association_cardinality_evaluability"
)


SECONDARY_OUTCOME = {
    "metric": (
        "metric_association_cardinality_evaluability"
    ),
    "numerator": (
        "n_uml_2_html_"
        "association_cardinalities_evaluable"
    ),
    "denominator": (
        "n_uml_2_html_"
        "association_ends_to_html"
    ),
}

def validate_secondary_source(
    canonical_data,
):
    required_columns = {
        "metric_generation_success",
        SECONDARY_OUTCOME["metric"],
        SECONDARY_OUTCOME["numerator"],
        SECONDARY_OUTCOME["denominator"],
    }

    missing_columns = (
        required_columns
        - set(canonical_data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Canonical dataset is missing "
            "secondary-outcome columns: "
            f"{sorted(missing_columns)}"
        )

    metric = canonical_data[
        SECONDARY_OUTCOME["metric"]
    ]

    numerator = canonical_data[
        SECONDARY_OUTCOME["numerator"]
    ]

    denominator = canonical_data[
        SECONDARY_OUTCOME["denominator"]
    ]

    generation_success = canonical_data[
        "metric_generation_success"
    ]

    failed = ~generation_success

    if metric[
        failed
    ].notna().any():
        raise ValueError(
            "association_cardinality_evaluability: "
            "failed generations contain "
            "defined metric values."
        )

    defined_metric = metric.notna()

    if numerator[
        defined_metric
    ].isna().any():
        raise ValueError(
            "association_cardinality_evaluability: "
            "defined metric with NA numerator."
        )

    if denominator[
        defined_metric
    ].isna().any():
        raise ValueError(
            "association_cardinality_evaluability: "
            "defined metric with NA denominator."
        )

    if (
        denominator[
            defined_metric
        ] <= 0
    ).any():
        raise ValueError(
            "association_cardinality_evaluability: "
            "defined metric with non-positive "
            "denominator."
        )

    defined_counts = (
        numerator.notna()
        & denominator.notna()
    )

    if (
        numerator[
            defined_counts
        ] < 0
    ).any():
        raise ValueError(
            "association_cardinality_evaluability: "
            "negative numerator."
        )

    if (
        denominator[
            defined_counts
        ] < 0
    ).any():
        raise ValueError(
            "association_cardinality_evaluability: "
            "negative denominator."
        )

    if (
        numerator[
            defined_counts
        ]
        > denominator[
            defined_counts
        ]
    ).any():
        raise ValueError(
            "association_cardinality_evaluability: "
            "numerator exceeds denominator."
        )

    
    reconstructed_metric = (
        numerator[
            defined_metric
        ]
        / denominator[
            defined_metric
        ]
    )

    if not np.allclose(
        metric[
            defined_metric
        ].to_numpy(
            dtype=float
        ),
        reconstructed_metric.to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=False,
    ):
        raise ValueError(
            "association_cardinality_evaluability: "
            "canonical metric does not match "
            "numerator / denominator."
        )

    evaluable_from_counts = (
        numerator.notna()
        & denominator.notna()
        & (denominator > 0)
    )

    missing_metric = (
        evaluable_from_counts
        & metric.isna()
    )

    if missing_metric.any():
        raise ValueError(
            "association_cardinality_evaluability: "
            "positive denominator and defined "
            "counts but metric is NA."
        )

def prepare_secondary_outcome_data(
    canonical_data,
):
    metric_column = (
        SECONDARY_OUTCOME["metric"]
    )

    numerator_column = (
        SECONDARY_OUTCOME["numerator"]
    )

    denominator_column = (
        SECONDARY_OUTCOME["denominator"]
    )

    evaluable = (
        canonical_data[
            "metric_generation_success"
        ]
        & canonical_data[
            metric_column
        ].notna()
        & canonical_data[
            numerator_column
        ].notna()
        & canonical_data[
            denominator_column
        ].notna()
        & (
            canonical_data[
                denominator_column
            ] > 0
        )
    )

    outcome_data = canonical_data.loc[
        evaluable,
        ID_COLUMNS
        + [
            "cell_id",
            metric_column,
            numerator_column,
            denominator_column,
        ],
    ].copy()

    outcome_data = outcome_data.rename(
        columns={
            metric_column: "metric",
            numerator_column: "numerator",
            denominator_column: "denominator",
        }
    )

    outcome_data["n_not_success"] = (
        outcome_data["denominator"]
        - outcome_data["numerator"]
    )

    return outcome_data


def validate_secondary_outcome_data(
    canonical_data,
    outcome_data,
):
    if outcome_data.empty:
        raise ValueError(
            "Secondary analytical dataset "
            "is empty."
        )

    expected_columns = {
        *ID_COLUMNS,
        "cell_id",
        "metric",
        "numerator",
        "denominator",
        "n_not_success",
    }

    missing_columns = (
        expected_columns
        - set(outcome_data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Secondary analytical dataset "
            "is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if outcome_data[
        ID_COLUMNS
    ].duplicated().any():
        raise ValueError(
            "Duplicate experimental attempts "
            "in secondary analytical dataset."
        )

    if outcome_data[
        "cell_id"
    ].isna().any():
        raise ValueError(
            "Secondary analytical dataset "
            "contains NA cell_id."
        )

    if (
        outcome_data[
            "denominator"
        ] <= 0
    ).any():
        raise ValueError(
            "Secondary analytical dataset "
            "contains non-positive denominator."
        )

    if (
        outcome_data[
            "numerator"
        ] < 0
    ).any():
        raise ValueError(
            "Secondary analytical dataset "
            "contains negative numerator."
        )

    if (
        outcome_data[
            "n_not_success"
        ] < 0
    ).any():
        raise ValueError(
            "Secondary analytical dataset "
            "contains numerator > denominator."
        )


    reconstructed_metric = (
        outcome_data[
            "numerator"
        ]
        / outcome_data[
            "denominator"
        ]
    )

    if not np.allclose(
        outcome_data[
            "metric"
        ].to_numpy(
            dtype=float
        ),
        reconstructed_metric.to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=False,
    ):
        raise ValueError(
            "Secondary analytical metric does "
            "not match numerator / denominator."
        )

    source = SECONDARY_OUTCOME

    expected_mask = (
        canonical_data[
            "metric_generation_success"
        ]
        & canonical_data[
            source["metric"]
        ].notna()
        & canonical_data[
            source["numerator"]
        ].notna()
        & canonical_data[
            source["denominator"]
        ].notna()
        & (
            canonical_data[
                source["denominator"]
            ] > 0
        )
    )

    expected_ids = (
        canonical_data.loc[
            expected_mask,
            ID_COLUMNS,
        ]
        .sort_values(
            ID_COLUMNS
        )
        .reset_index(
            drop=True
        )
    )

    observed_ids = (
        outcome_data[
            ID_COLUMNS
        ]
        .sort_values(
            ID_COLUMNS
        )
        .reset_index(
            drop=True
        )
    )

    if not expected_ids.equals(
        observed_ids
    ):
        raise ValueError(
            "Secondary analytical rows do "
            "not exactly match the canonical "
            "eligible attempts."
        )

    cell_sizes = (
        outcome_data
        .groupby(
            "cell_id",
            dropna=False,
        )
        .size()
    )

    if (
        cell_sizes > 10
    ).any():
        raise ValueError(
            "A secondary analytical cell "
            "contains more than 10 attempts."
        )

def write_secondary_data(
    outcome_data,
):
    SECONDARY_ANALYSIS_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        SECONDARY_ANALYSIS_DATA_PATH
        / (
            SECONDARY_OUTCOME_NAME
            + ".csv"
        )
    )

    outcome_data.to_csv(
        filename,
        index=False,
    )

    return filename

def summarize_secondary_data(
    outcome_data,
):
    cell_sizes = (
        outcome_data
        .groupby(
            "cell_id"
        )
        .size()
    )

    denominator_nunique = (
        outcome_data
        .groupby(
            "cell_id"
        )[
            "denominator"
        ]
        .nunique()
    )

    numerator_nunique = (
        outcome_data
        .groupby(
            "cell_id"
        )[
            "numerator"
        ]
        .nunique()
    )

    return {
        "attempt_rows": int(
            len(outcome_data)
        ),
        "cells": int(
            outcome_data[
                "cell_id"
            ].nunique()
        ),
        "min_attempts_per_cell": int(
            cell_sizes.min()
        ),
        "max_attempts_per_cell": int(
            cell_sizes.max()
        ),
        "constant_numerator_cells": int(
            (
                numerator_nunique == 1
            ).sum()
        ),
        "variable_numerator_cells": int(
            (
                numerator_nunique > 1
            ).sum()
        ),
        "variable_denominator_cells": int(
            (
                denominator_nunique > 1
            ).sum()
        ),
        "metric_min": float(
            outcome_data[
                "metric"
            ].min()
        ),
        "metric_max": float(
            outcome_data[
                "metric"
            ].max()
        ),
    }

def main():
    canonical_data = (
        load_canonical_dataset()
    )

    validate_canonical_dataset(
        canonical_data
    )

    validate_secondary_source(
        canonical_data
    )

    canonical_data = add_cell_id(
        canonical_data
    )

    outcome_data = (
        prepare_secondary_outcome_data(
            canonical_data
        )
    )

    validate_secondary_outcome_data(
        canonical_data,
        outcome_data,
    )

    summary = (
        summarize_secondary_data(
            outcome_data
        )
    )

    filename = (
        write_secondary_data(
            outcome_data
        )
    )

    print(
        "Secondary analytical dataset "
        "prepared and validated successfully."
    )

    print(
        "outcome: "
        f"{SECONDARY_OUTCOME_NAME}"
    )

    for key, value in (
        summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        f"saved: {filename}"
    )


if __name__ == "__main__":
    main()