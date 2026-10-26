from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_DATASET_FILENAME = Path(
    "experiment/metrics/canonical_dataset.csv"
)

ANALYSIS_DATA_PATH = Path(
    "experiment/analysis/data"
)


ID_COLUMNS = [
    "model",
    "uml_model",
    "representation",
    "iteration",
]


CELL_COLUMNS = [
    "model",
    "uml_model",
    "representation",
]


PRIMARY_OUTCOMES = {
    "attribute_coverage": {
        "metric": "metric_attribute_coverage",
        "numerator": "n_uml_2_html_class_attributes_to_html",
        "denominator": "n_uml_attributes",
    },
    "primitive_type_fidelity": {
        "metric": "metric_primitive_type_fidelity",
        "numerator": "n_uml_2_html_primitive_types_preserved",
        "denominator": "n_uml_2_html_primitive_types_evaluable",
    },
    "enumeration_literal_coverage": {
        "metric": "metric_enumeration_literal_coverage",
        "numerator": "n_uml_2_html_enumeration_literals_to_html",
        "denominator": "n_uml_enumeration_attribute_literals",
    },
    "enumeration_cardinality_fidelity": {
        "metric": "metric_enumeration_cardinality_fidelity",
        "numerator": (
            "n_uml_2_html_"
            "enumeration_cardinalities_preserved"
        ),
        "denominator": (
            "n_uml_2_html_"
            "enumeration_cardinalities_evaluable"
        ),
    },
    "attribute_nonempty_label_coverage": {
        "metric": "metric_attribute_nonempty_label_coverage",
        "numerator": (
            "n_uml_2_html_"
            "attributes_with_nonempty_label"
        ),
        "denominator": (
            "n_uml_2_html_"
            "class_attributes_to_html"
        ),
    },
    "association_end_coverage": {
        "metric": "metric_association_end_coverage",
        "numerator": (
            "n_uml_2_html_"
            "association_ends_to_html"
        ),
        "denominator": "n_uml_ends",
    },
    "association_cardinality_fidelity": {
        "metric": "metric_association_cardinality_fidelity",
        "numerator": (
            "n_uml_2_html_"
            "association_cardinalities_preserved"
        ),
        "denominator": (
            "n_uml_2_html_"
            "association_cardinalities_evaluable"
        ),
    },
}

def load_canonical_dataset():
    if not CANONICAL_DATASET_FILENAME.is_file():
        raise FileNotFoundError(
            "Canonical dataset not found: "
            f"{CANONICAL_DATASET_FILENAME}"
        )

    return pd.read_csv(
        CANONICAL_DATASET_FILENAME
    )


def validate_canonical_dataset(dataframe):
    required_columns = set(
        ID_COLUMNS
        + [
            "metric_generation_success",
        ]
    )

    for outcome in PRIMARY_OUTCOMES.values():
        required_columns.update(
            [
                outcome["metric"],
                outcome["numerator"],
                outcome["denominator"],
            ]
        )

    missing_columns = sorted(
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns in "
            "canonical dataset: "
            f"{missing_columns}"
        )

    if len(dataframe) != 1500:
        raise ValueError(
            "Expected 1500 experimental attempts, "
            f"found {len(dataframe)}."
        )

    duplicated = dataframe.duplicated(
        subset=ID_COLUMNS,
        keep=False,
    )

    if duplicated.any():
        duplicated_ids = dataframe.loc[
            duplicated,
            ID_COLUMNS,
        ]

        raise ValueError(
            "Duplicate experimental attempt "
            "identifiers found:\n"
            + duplicated_ids.to_string(
                index=False
            )
        )

    expected_models = {
        "llama3.1:8b",
        "mistral:7b-instruct",
        "qwen2.5:7b-instruct",
    }

    observed_models = set(
        dataframe["model"].unique()
    )

    if observed_models != expected_models:
        raise ValueError(
            "Unexpected LLM set. "
            f"Expected {sorted(expected_models)}, "
            f"found {sorted(observed_models)}."
        )

    expected_uml_models = {
        f"uml{i}"
        for i in range(1, 11)
    }

    observed_uml_models = set(
        dataframe["uml_model"].unique()
    )

    if (
        observed_uml_models
        != expected_uml_models
    ):
        raise ValueError(
            "Unexpected UML-model set. "
            f"Expected "
            f"{sorted(expected_uml_models)}, "
            f"found "
            f"{sorted(observed_uml_models)}."
        )

    expected_representations = {
        "uml",
        "json",
        "xsd",
        "sql",
        "txt",
    }

    observed_representations = set(
        dataframe[
            "representation"
        ].unique()
    )

    if (
        observed_representations
        != expected_representations
    ):
        raise ValueError(
            "Unexpected representation set. "
            f"Expected "
            f"{sorted(expected_representations)}, "
            f"found "
            f"{sorted(observed_representations)}."
        )

    expected_iterations = set(range(10))

    observed_iterations = set(
        dataframe["iteration"].unique()
    )

    if (
        observed_iterations
        != expected_iterations
    ):
        raise ValueError(
            "Unexpected iteration set. "
            f"Expected "
            f"{sorted(expected_iterations)}, "
            f"found "
            f"{sorted(observed_iterations)}."
        )

    cell_sizes = (
        dataframe
        .groupby(
            CELL_COLUMNS,
            dropna=False,
        )
        .size()
    )

    if len(cell_sizes) != 150:
        raise ValueError(
            "Expected 150 experimental cells, "
            f"found {len(cell_sizes)}."
        )

    invalid_cell_sizes = (
        cell_sizes != 10
    )

    if invalid_cell_sizes.any():
        raise ValueError(
            "Every experimental cell must "
            "contain exactly 10 attempts.\n"
            + cell_sizes[
                invalid_cell_sizes
            ].to_string()
        )

    generation_success = dataframe[
        "metric_generation_success"
    ]

    if generation_success.isna().any():
        raise ValueError(
            "metric_generation_success "
            "contains NA values."
        )

    valid_generation_values = (
        generation_success.isin(
            [True, False]
        )
    )

    if not valid_generation_values.all():
        raise ValueError(
            "metric_generation_success "
            "contains values other than "
            "True/False."
        )

    n_success = int(
        generation_success.sum()
    )

    n_not_success = (
        len(dataframe) - n_success
    )

    if n_success != 1479:
        raise ValueError(
            "Expected 1479 successful "
            "generations, "
            f"found {n_success}."
        )

    if n_not_success != 21:
        raise ValueError(
            "Expected 21 failed "
            "generations, "
            f"found {n_not_success}."
        )

def validate_primary_outcomes(dataframe):
    generation_success = dataframe[
        "metric_generation_success"
    ]

    for outcome_name, outcome in PRIMARY_OUTCOMES.items():
        metric_column = outcome["metric"]
        numerator_column = outcome["numerator"]
        denominator_column = outcome["denominator"]

        metric = dataframe[metric_column]
        numerator = dataframe[numerator_column]
        denominator = dataframe[denominator_column]

        failed = ~generation_success

        # A failed generation must not contain an evaluable
        # semantic metric.
        if metric[failed].notna().any():
            raise ValueError(
                f"{outcome_name}: failed generations contain "
                "defined metric values."
            )

        # The numerator and denominator used by the statistical
        # analysis must be defined whenever the metric is defined.
        defined_metric = metric.notna()

        if numerator[defined_metric].isna().any():
            raise ValueError(
                f"{outcome_name}: defined metric with NA numerator."
            )

        if denominator[defined_metric].isna().any():
            raise ValueError(
                f"{outcome_name}: defined metric with NA denominator."
            )

        # A defined ratio requires a strictly positive denominator.
        if (denominator[defined_metric] <= 0).any():
            raise ValueError(
                f"{outcome_name}: defined metric with "
                "non-positive denominator."
            )

        # Count-based metrics cannot have negative counts.
        defined_counts = (
            numerator.notna()
            & denominator.notna()
        )

        if (numerator[defined_counts] < 0).any():
            raise ValueError(
                f"{outcome_name}: negative numerator found."
            )

        if (denominator[defined_counts] < 0).any():
            raise ValueError(
                f"{outcome_name}: negative denominator found."
            )

        # The number of successes cannot exceed the number
        # of eligible/evaluable elements.
        if (
            numerator[defined_counts]
            > denominator[defined_counts]
        ).any():
            raise ValueError(
                f"{outcome_name}: numerator exceeds denominator."
            )

        # Reconstruct the metric independently from its source
        # counts and compare it with the canonical ratio.
        reconstructed_metric = (
            numerator[defined_metric]
            / denominator[defined_metric]
        )

        if not np.allclose(
            metric[defined_metric].to_numpy(
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
                f"{outcome_name}: canonical metric does not "
                "match numerator / denominator."
            )

        # If both source counts exist and the denominator is
        # positive, the metric must be defined.
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
                f"{outcome_name}: positive denominator and "
                "defined counts but metric is NA."
            )

def add_cell_id(dataframe):
    result = dataframe.copy()

    if result[CELL_COLUMNS].isna().any().any():
        raise ValueError(
            "Cannot construct cell_id because one or more "
            "cell identifier columns contain NA."
        )

    result["cell_id"] = (
        result["model"].astype(str)
        + "__"
        + result["uml_model"].astype(str)
        + "__"
        + result["representation"].astype(str)
    )

    if result["cell_id"].nunique() != 150:
        raise ValueError(
            "Expected 150 unique cell_id values, "
            f"found {result['cell_id'].nunique()}."
        )

    cell_id_sizes = (
        result
        .groupby(
            "cell_id",
            dropna=False,
        )
        .size()
    )

    if not (cell_id_sizes == 10).all():
        raise ValueError(
            "Every cell_id must contain exactly "
            "10 experimental attempts."
        )

    return result

def prepare_generation_success_data(dataframe):
    generation_data = (
        dataframe
        .groupby(
            CELL_COLUMNS,
            as_index=False,
            dropna=False,
        )
        .agg(
            n_success=(
                "metric_generation_success",
                "sum",
            ),
            n_trials=(
                "metric_generation_success",
                "size",
            ),
        )
    )

    generation_data["n_not_success"] = (
        generation_data["n_trials"]
        - generation_data["n_success"]
    )

    generation_data["success_rate"] = (
        generation_data["n_success"]
        / generation_data["n_trials"]
    )

    if len(generation_data) != 150:
        raise ValueError(
            "Generation-success dataset must contain "
            f"150 cells, found {len(generation_data)}."
        )

    if not (generation_data["n_trials"] == 10).all():
        raise ValueError(
            "Every generation-success cell must contain "
            "exactly 10 trials."
        )

    if (
        generation_data["n_success"].sum()
        != 1479
    ):
        raise ValueError(
            "Generation-success aggregation does not "
            "preserve the expected 1479 successes."
        )

    if (
        generation_data["n_not_success"].sum()
        != 21
    ):
        raise ValueError(
            "Generation-success aggregation does not "
            "preserve the expected 21 failures."
        )

    return generation_data


def prepare_primary_outcome_data(
    dataframe,
    outcome_name,
):
    if outcome_name not in PRIMARY_OUTCOMES:
        raise KeyError(
            f"Unknown primary outcome: {outcome_name}"
        )

    outcome = PRIMARY_OUTCOMES[outcome_name]

    metric_column = outcome["metric"]
    numerator_column = outcome["numerator"]
    denominator_column = outcome["denominator"]

    evaluable = (
        dataframe["metric_generation_success"]
        & dataframe[metric_column].notna()
        & dataframe[numerator_column].notna()
        & dataframe[denominator_column].notna()
        & (dataframe[denominator_column] > 0)
    )

    outcome_data = dataframe.loc[
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

    if outcome_data.empty:
        raise ValueError(
            f"{outcome_name}: analytical dataset is empty."
        )

    if outcome_data[ID_COLUMNS].duplicated().any():
        raise ValueError(
            f"{outcome_name}: duplicate experimental "
            "attempts found in analytical dataset."
        )

    if (outcome_data["denominator"] <= 0).any():
        raise ValueError(
            f"{outcome_name}: non-positive denominator "
            "found after filtering."
        )

    if (outcome_data["numerator"] < 0).any():
        raise ValueError(
            f"{outcome_name}: negative numerator "
            "found after filtering."
        )

    if (outcome_data["n_not_success"] < 0).any():
        raise ValueError(
            f"{outcome_name}: numerator exceeds denominator "
            "after filtering."
        )

    reconstructed_metric = (
        outcome_data["numerator"]
        / outcome_data["denominator"]
    )

    if not np.allclose(
        outcome_data["metric"].to_numpy(
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
            f"{outcome_name}: analytical metric does not "
            "match numerator / denominator."
        )

    return outcome_data


EXPECTED_OUTCOME_ROWS = {
    "attribute_coverage": 1479,
    "primitive_type_fidelity": 1389,
    "enumeration_literal_coverage": 1040,
    "enumeration_cardinality_fidelity": 803,
    "attribute_nonempty_label_coverage": 1389,
    "association_end_coverage": 1479,
    "association_cardinality_fidelity": 971,
}


def validate_analysis_data(
    canonical_data,
    generation_data,
    outcome_datasets,
):
    if len(generation_data) != 150:
        raise ValueError(
            "Expected 150 generation-success cells, "
            f"found {len(generation_data)}."
        )

    expected_generation_columns = {
        *CELL_COLUMNS,
        "n_success",
        "n_trials",
        "n_not_success",
        "success_rate",
    }

    missing_generation_columns = (
        expected_generation_columns
        - set(generation_data.columns)
    )

    if missing_generation_columns:
        raise KeyError(
            "Generation-success analytical dataset is "
            "missing columns: "
            f"{sorted(missing_generation_columns)}"
        )

    if set(outcome_datasets) != set(PRIMARY_OUTCOMES):
        raise ValueError(
            "Analytical outcome datasets do not match "
            "the configured primary outcomes."
        )

    for outcome_name, outcome_data in outcome_datasets.items():
        expected_rows = EXPECTED_OUTCOME_ROWS[outcome_name]

        if len(outcome_data) != expected_rows:
            raise ValueError(
                f"{outcome_name}: expected {expected_rows} "
                f"analytical rows, found {len(outcome_data)}."
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
                f"{outcome_name}: analytical dataset is "
                "missing columns: "
                f"{sorted(missing_columns)}"
            )

        if outcome_data["cell_id"].isna().any():
            raise ValueError(
                f"{outcome_name}: cell_id contains NA."
            )

        if outcome_data["cell_id"].nunique() > 150:
            raise ValueError(
                f"{outcome_name}: more than 150 "
                "experimental cells found."
            )

        cell_sizes = (
            outcome_data
            .groupby(
                "cell_id",
                dropna=False,
            )
            .size()
        )

        if (cell_sizes > 10).any():
            raise ValueError(
                f"{outcome_name}: an analytical cell "
                "contains more than 10 attempts."
            )

        source = PRIMARY_OUTCOMES[outcome_name]

        expected_mask = (
            canonical_data["metric_generation_success"]
            & canonical_data[source["metric"]].notna()
            & canonical_data[source["numerator"]].notna()
            & canonical_data[source["denominator"]].notna()
            & (canonical_data[source["denominator"]] > 0)
        )

        expected_ids = (
            canonical_data.loc[
                expected_mask,
                ID_COLUMNS,
            ]
            .sort_values(ID_COLUMNS)
            .reset_index(drop=True)
        )

        observed_ids = (
            outcome_data[ID_COLUMNS]
            .sort_values(ID_COLUMNS)
            .reset_index(drop=True)
        )

        if not expected_ids.equals(observed_ids):
            raise ValueError(
                f"{outcome_name}: analytical rows do not "
                "match the evaluable canonical attempts."
            )


def write_analysis_data(
    generation_data,
    outcome_datasets,
):
    ANALYSIS_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    generation_data.to_csv(
        ANALYSIS_DATA_PATH
        / "generation_success.csv",
        index=False,
    )

    for outcome_name, outcome_data in outcome_datasets.items():
        outcome_data.to_csv(
            ANALYSIS_DATA_PATH
            / f"{outcome_name}.csv",
            index=False,
        )


def main():
    canonical_data = load_canonical_dataset()

    validate_canonical_dataset(
        canonical_data
    )

    validate_primary_outcomes(
        canonical_data
    )

    canonical_data = add_cell_id(
        canonical_data
    )

    generation_data = (
        prepare_generation_success_data(
            canonical_data
        )
    )

    outcome_datasets = {}

    for outcome_name in PRIMARY_OUTCOMES:
        outcome_datasets[outcome_name] = (
            prepare_primary_outcome_data(
                canonical_data,
                outcome_name,
            )
        )

    validate_analysis_data(
        canonical_data,
        generation_data,
        outcome_datasets,
    )

    write_analysis_data(
        generation_data,
        outcome_datasets,
    )

    print(
        "Analysis datasets prepared and validated "
        "successfully."
    )

    print(
        "generation_success: "
        f"{len(generation_data)} rows"
    )

    for outcome_name, outcome_data in outcome_datasets.items():
        print(
            f"{outcome_name}: "
            f"{len(outcome_data)} rows"
        )


if __name__ == "__main__":
    main()