from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILENAME = Path(
    "experiment/analysis/data/secondary/cell_level/"
    "association_cardinality_evaluability.csv"
)

RESULTS_PATH = Path(
    "experiment/analysis/results/secondary/"
    "association_cardinality_evaluability"
)

RESULTS_FILENAME = (
    RESULTS_PATH
    / "contrasts.csv"
)


OUTCOME_NAME = (
    "association_cardinality_evaluability"
)


MODEL_LEVELS = [
    "llama3.1:8b",
    "mistral:7b-instruct",
    "qwen2.5:7b-instruct",
]


REFERENCE_REPRESENTATION = "uml"


COMPARISON_REPRESENTATIONS = [
    "json",
    "xsd",
    "sql",
    "txt",
]


UML_LEVELS = [
    f"uml{i}"
    for i in range(1, 11)
]


BAYESIAN_BOOTSTRAP_DRAWS = 20_000
RANDOM_SEED = 1977

ZERO_TOLERANCE = 1e-15

EXPECTED_CELLS = 108
EXPECTED_CONTRASTS = 12

def load_data():
    if not INPUT_FILENAME.is_file():
        raise FileNotFoundError(
            f"Secondary cell-level dataset "
            f"not found: {INPUT_FILENAME}"
        )

    return pd.read_csv(
        INPUT_FILENAME
    )


def validate_data(
    dataframe,
):
    required_columns = {
        "cell_id",
        "model",
        "uml_model",
        "representation",
        "n_attempts",
        "numerator_sum",
        "denominator_sum",
        "numerator_mean",
        "denominator_mean",
        "cell_ratio",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Secondary cell-level dataset "
            "is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != EXPECTED_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_CELLS} cells, "
            f"found {len(dataframe)}."
        )

    if dataframe[
        [
            "model",
            "uml_model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate LLM × UML × "
            "representation cells."
        )

    if (
        dataframe["denominator_sum"] <= 0
    ).any():
        raise ValueError(
            "denominator_sum must be positive."
        )

    if (
        dataframe["denominator_mean"] <= 0
    ).any():
        raise ValueError(
            "denominator_mean must be positive."
        )

    ratio_from_sums = (
        dataframe["numerator_sum"]
        / dataframe["denominator_sum"]
    )

    if not np.allclose(
        dataframe["cell_ratio"],
        ratio_from_sums,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "cell_ratio does not match "
            "numerator_sum / denominator_sum."
        )

    ratio_from_means = (
        dataframe["numerator_mean"]
        / dataframe["denominator_mean"]
    )

    if not np.allclose(
        dataframe["cell_ratio"],
        ratio_from_means,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "cell_ratio does not match "
            "numerator_mean / denominator_mean."
        )

    if not set(
        dataframe["model"]
    ).issubset(
        set(MODEL_LEVELS)
    ):
        raise ValueError(
            "Unexpected model."
        )

    valid_representations = {
        REFERENCE_REPRESENTATION,
        *COMPARISON_REPRESENTATIONS,
    }

    if not set(
        dataframe["representation"]
    ).issubset(
        valid_representations
    ):
        raise ValueError(
            "Unexpected representation."
        )


def sort_uml_models(
    values,
):
    return sorted(
        values,
        key=lambda value: int(
            value.replace("uml", "")
        ),
    )


def compute_weighted_estimands(
    dataframe,
    weights,
):
    numerator_mean = dataframe[
        "numerator_mean"
    ].to_numpy(
        dtype=float
    )

    denominator_mean = dataframe[
        "denominator_mean"
    ].to_numpy(
        dtype=float
    )

    cell_ratio = dataframe[
        "cell_ratio"
    ].to_numpy(
        dtype=float
    )

    element_weighted = (
        np.sum(
            weights * numerator_mean,
            axis=-1,
        )
        / np.sum(
            weights * denominator_mean,
            axis=-1,
        )
    )

    model_weighted = np.sum(
        weights * cell_ratio,
        axis=-1,
    )

    return (
        element_weighted,
        model_weighted,
    )

def prepare_paired_contrast(
    dataframe,
    model_name,
    comparison_name,
):
    model_data = dataframe[
        dataframe["model"]
        == model_name
    ]

    reference = model_data[
        model_data["representation"]
        == REFERENCE_REPRESENTATION
    ].copy()

    comparison = model_data[
        model_data["representation"]
        == comparison_name
    ].copy()

    paired_umls = (
        set(reference["uml_model"])
        & set(comparison["uml_model"])
    )

    uml_order = sort_uml_models(
        paired_umls
    )

    if not uml_order:
        return (
            reference.iloc[0:0].copy(),
            comparison.iloc[0:0].copy(),
            uml_order,
        )

    reference = (
        reference[
            reference["uml_model"].isin(
                paired_umls
            )
        ]
        .set_index("uml_model")
        .loc[uml_order]
        .reset_index()
    )

    comparison = (
        comparison[
            comparison["uml_model"].isin(
                paired_umls
            )
        ]
        .set_index("uml_model")
        .loc[uml_order]
        .reset_index()
    )

    if not np.array_equal(
        reference[
            "uml_model"
        ].to_numpy(),
        comparison[
            "uml_model"
        ].to_numpy(),
    ):
        raise ValueError(
            "Reference and comparison UMLs "
            "are not aligned."
        )

    return (
        reference,
        comparison,
        uml_order,
    )

def compute_uniform_weight_diagnostics(
    reference,
    comparison,
):
    n_umls = len(reference)

    if n_umls == 0:
        raise ValueError(
            "Uniform diagnostics require "
            "at least one paired UML."
        )

    weights = np.full(
        n_umls,
        1.0 / n_umls,
        dtype=float,
    )

    (
        reference_element,
        reference_model,
    ) = compute_weighted_estimands(
        reference,
        weights,
    )

    (
        comparison_element,
        comparison_model,
    ) = compute_weighted_estimands(
        comparison,
        weights,
    )

    element_difference = (
        comparison_element
        - reference_element
    )

    model_difference = (
        comparison_model
        - reference_model
    )

    direct_reference_element = (
        reference[
            "numerator_mean"
        ].sum()
        / reference[
            "denominator_mean"
        ].sum()
    )

    direct_comparison_element = (
        comparison[
            "numerator_mean"
        ].sum()
        / comparison[
            "denominator_mean"
        ].sum()
    )

    direct_element_difference = (
        direct_comparison_element
        - direct_reference_element
    )

    direct_model_difference = (
        comparison[
            "cell_ratio"
        ].mean()
        - reference[
            "cell_ratio"
        ].mean()
    )

    if not np.isclose(
        element_difference,
        direct_element_difference,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Uniform element-weighted "
            "contrast failed validation."
        )

    if not np.isclose(
        model_difference,
        direct_model_difference,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Uniform model-weighted "
            "contrast failed validation."
        )

    return {
        "reference_element_weighted": float(
            reference_element
        ),
        "comparison_element_weighted": float(
            comparison_element
        ),
        "element_weighted_difference": float(
            element_difference
        ),
        "reference_model_weighted": float(
            reference_model
        ),
        "comparison_model_weighted": float(
            comparison_model
        ),
        "model_weighted_difference": float(
            model_difference
        ),
    }

def generate_bayesian_bootstrap_weights(
    n_units,
):
    if n_units < 2:
        raise ValueError(
            "Bayesian bootstrap requires "
            "at least two paired UML models."
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    weights = rng.dirichlet(
        alpha=np.ones(
            n_units,
            dtype=float,
        ),
        size=BAYESIAN_BOOTSTRAP_DRAWS,
    )

    expected_shape = (
        BAYESIAN_BOOTSTRAP_DRAWS,
        n_units,
    )

    if weights.shape != expected_shape:
        raise ValueError(
            "Unexpected bootstrap weight shape."
        )

    if not np.allclose(
        weights.sum(axis=1),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Bootstrap weights do not "
            "sum to one."
        )

    if (
        weights <= 0
    ).any():
        raise ValueError(
            "Bootstrap weights must be "
            "strictly positive."
        )

    return weights

def compute_bootstrap_contrasts(
    reference,
    comparison,
    weights,
):
    (
        reference_element,
        reference_model,
    ) = compute_weighted_estimands(
        reference,
        weights,
    )

    (
        comparison_element,
        comparison_model,
    ) = compute_weighted_estimands(
        comparison,
        weights,
    )

    element_difference = (
        comparison_element
        - reference_element
    )

    model_difference = (
        comparison_model
        - reference_model
    )

    if not np.isfinite(
        element_difference
    ).all():
        raise ValueError(
            "Non-finite element-weighted "
            "bootstrap differences."
        )

    if not np.isfinite(
        model_difference
    ).all():
        raise ValueError(
            "Non-finite model-weighted "
            "bootstrap differences."
        )

    return (
        element_difference,
        model_difference,
    )

def summarize_draws(
    draws,
):
    positive = (
        draws > ZERO_TOLERANCE
    )

    negative = (
        draws < -ZERO_TOLERANCE
    )

    zero = (
        ~(positive | negative)
    )

    result = {
        "mean": float(
            np.mean(draws)
        ),
        "median": float(
            np.median(draws)
        ),
        "q025": float(
            np.quantile(
                draws,
                0.025,
            )
        ),
        "q975": float(
            np.quantile(
                draws,
                0.975,
            )
        ),
        "probability_gt_0": float(
            np.mean(positive)
        ),
        "probability_lt_0": float(
            np.mean(negative)
        ),
        "probability_eq_0": float(
            np.mean(zero)
        ),
    }

    probability_sum = (
        result["probability_gt_0"]
        + result["probability_lt_0"]
        + result["probability_eq_0"]
    )

    if not np.isclose(
        probability_sum,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Direction probabilities "
            "do not sum to one."
        )

    return result

def add_empty_bootstrap_summary(
    result,
):
    columns = [
        "element_bb_mean",
        "element_bb_median",
        "element_bb_q025",
        "element_bb_q975",
        "element_bb_probability_gt_0",
        "element_bb_probability_lt_0",
        "element_bb_probability_eq_0",
        "model_weighted_bb_mean",
        "model_weighted_bb_median",
        "model_weighted_bb_q025",
        "model_weighted_bb_q975",
        "model_weighted_bb_probability_gt_0",
        "model_weighted_bb_probability_lt_0",
        "model_weighted_bb_probability_eq_0",
    ]

    for column in columns:
        result[column] = np.nan

def analyze_contrast(
    dataframe,
    model_name,
    comparison_name,
):
    (
        reference,
        comparison,
        paired_umls,
    ) = prepare_paired_contrast(
        dataframe,
        model_name,
        comparison_name,
    )

    n_paired = len(
        paired_umls
    )

    result = {
        "outcome": OUTCOME_NAME,
        "model": model_name,
        "reference": (
            REFERENCE_REPRESENTATION
        ),
        "comparison": comparison_name,
        "n_paired_umls": n_paired,
        "paired_umls": ",".join(
            paired_umls
        ),
    }

    if n_paired == 0:
        result[
            "analysis_status"
        ] = "not_estimable"

        for column in [
            "reference_element_weighted",
            "comparison_element_weighted",
            "element_weighted_difference",
            "reference_model_weighted",
            "comparison_model_weighted",
            "model_weighted_difference",
        ]:
            result[column] = np.nan

        add_empty_bootstrap_summary(
            result
        )

        return result

    uniform = (
        compute_uniform_weight_diagnostics(
            reference,
            comparison,
        )
    )

    result.update(
        uniform
    )

    if n_paired == 1:
        result[
            "analysis_status"
        ] = "descriptive_only"

        add_empty_bootstrap_summary(
            result
        )

        return result

    result[
        "analysis_status"
    ] = "bayesian_bootstrap"

    weights = (
        generate_bayesian_bootstrap_weights(
            n_paired
        )
    )

    (
        element_draws,
        model_draws,
    ) = compute_bootstrap_contrasts(
        reference,
        comparison,
        weights,
    )

    element_summary = summarize_draws(
        element_draws
    )

    model_summary = summarize_draws(
        model_draws
    )

    result.update(
        {
            "element_bb_mean": (
                element_summary["mean"]
            ),
            "element_bb_median": (
                element_summary["median"]
            ),
            "element_bb_q025": (
                element_summary["q025"]
            ),
            "element_bb_q975": (
                element_summary["q975"]
            ),
            "element_bb_probability_gt_0": (
                element_summary[
                    "probability_gt_0"
                ]
            ),
            "element_bb_probability_lt_0": (
                element_summary[
                    "probability_lt_0"
                ]
            ),
            "element_bb_probability_eq_0": (
                element_summary[
                    "probability_eq_0"
                ]
            ),
            "model_weighted_bb_mean": (
                model_summary["mean"]
            ),
            "model_weighted_bb_median": (
                model_summary["median"]
            ),
            "model_weighted_bb_q025": (
                model_summary["q025"]
            ),
            "model_weighted_bb_q975": (
                model_summary["q975"]
            ),
            "model_weighted_bb_probability_gt_0": (
                model_summary[
                    "probability_gt_0"
                ]
            ),
            "model_weighted_bb_probability_lt_0": (
                model_summary[
                    "probability_lt_0"
                ]
            ),
            "model_weighted_bb_probability_eq_0": (
                model_summary[
                    "probability_eq_0"
                ]
            ),
        }
    )

    return result

def validate_results(
    results,
):
    if len(results) != EXPECTED_CONTRASTS:
        raise ValueError(
            f"Expected {EXPECTED_CONTRASTS} "
            f"secondary contrasts, found "
            f"{len(results)}."
        )

    if results[
        [
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate secondary contrasts."
        )

    if not (
        results["outcome"]
        == OUTCOME_NAME
    ).all():
        raise ValueError(
            "Unexpected secondary outcome."
        )

    if (
        results.loc[
            results[
                "n_paired_umls"
            ] == 0,
            "analysis_status",
        ]
        != "not_estimable"
    ).any():
        raise ValueError(
            "n=0 contrast has invalid status."
        )

    if (
        results.loc[
            results[
                "n_paired_umls"
            ] == 1,
            "analysis_status",
        ]
        != "descriptive_only"
    ).any():
        raise ValueError(
            "n=1 contrast has invalid status."
        )

    if (
        results.loc[
            results[
                "n_paired_umls"
            ] >= 2,
            "analysis_status",
        ]
        != "bayesian_bootstrap"
    ).any():
        raise ValueError(
            "Inferential contrast has "
            "invalid status."
        )

def main():
    dataframe = load_data()

    validate_data(
        dataframe
    )

    rows = []

    for model_name in MODEL_LEVELS:
        for comparison_name in (
            COMPARISON_REPRESENTATIONS
        ):
            rows.append(
                analyze_contrast(
                    dataframe,
                    model_name,
                    comparison_name,
                )
            )

    results = pd.DataFrame(
        rows
    )

    validate_results(
        results
    )

    RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_FILENAME,
        index=False,
    )

    print(
        "Secondary association-cardinality "
        "evaluability Bayesian bootstrap "
        "completed and validated successfully."
    )

    print(
        f"contrasts: {len(results)}"
    )

    print()

    print(
        "analysis_status:"
    )

    print(
        results[
            "analysis_status"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "support:"
    )

    print(
        results[
            [
                "model",
                "comparison",
                "n_paired_umls",
                "analysis_status",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "element-weighted results:"
    )

    print(
        results[
            [
                "model",
                "comparison",
                "n_paired_umls",
                "analysis_status",
                "element_weighted_difference",
                "element_bb_mean",
                "element_bb_q025",
                "element_bb_q975",
                "element_bb_probability_gt_0",
                "element_bb_probability_lt_0",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"saved: {RESULTS_FILENAME}"
    )


if __name__ == "__main__":
    main()

