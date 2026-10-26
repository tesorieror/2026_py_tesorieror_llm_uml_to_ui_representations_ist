from pathlib import Path

import numpy as np
import pandas as pd


CELL_LEVEL_DATA_PATH = Path(
    "experiment/analysis/data/cell_level"
)


PILOT_OUTCOME = "attribute_coverage"
PILOT_MODEL = "llama3.1:8b"
REFERENCE_REPRESENTATION = "uml"
COMPARISON_REPRESENTATION = "json"

EXPECTED_PAIRED_UMLS = 10

BAYESIAN_BOOTSTRAP_DRAWS = 20_000
RANDOM_SEED = 1977


UML_LEVELS = [
    f"uml{i}"
    for i in range(1, 11)
]

def load_pilot_data():
    filename = (
        CELL_LEVEL_DATA_PATH
        / f"{PILOT_OUTCOME}.csv"
    )

    if not filename.is_file():
        raise FileNotFoundError(
            f"Cell-level dataset not found: "
            f"{filename}"
        )

    return pd.read_csv(filename)

def validate_pilot_data(
    dataframe,
):
    required_columns = {
        "model",
        "uml_model",
        "representation",
        "cell_id",
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
            "Pilot cell-level dataset is missing "
            f"columns: {sorted(missing_columns)}"
        )

    if dataframe[
        [
            "model",
            "uml_model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate cell-level observations."
        )

    if (
        dataframe["denominator_sum"] <= 0
    ).any():
        raise ValueError(
            "Cell denominator_sum must be positive."
        )

    if (
        dataframe["denominator_mean"] <= 0
    ).any():
        raise ValueError(
            "Cell denominator_mean must be positive."
        )

    reconstructed_ratio = (
        dataframe["numerator_sum"]
        / dataframe["denominator_sum"]
    )

    if not np.allclose(
        dataframe["cell_ratio"],
        reconstructed_ratio,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "cell_ratio does not match "
            "numerator_sum / denominator_sum."
        )

    reconstructed_from_means = (
        dataframe["numerator_mean"]
        / dataframe["denominator_mean"]
    )

    if not np.allclose(
        dataframe["cell_ratio"],
        reconstructed_from_means,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "cell_ratio does not match "
            "numerator_mean / denominator_mean."
        )

def prepare_paired_contrast(
    dataframe,
):
    pilot = dataframe[
        dataframe["model"]
        == PILOT_MODEL
    ].copy()

    reference = pilot[
        pilot["representation"]
        == REFERENCE_REPRESENTATION
    ].copy()

    comparison = pilot[
        pilot["representation"]
        == COMPARISON_REPRESENTATION
    ].copy()

    reference_umls = set(
        reference["uml_model"]
    )

    comparison_umls = set(
        comparison["uml_model"]
    )

    paired_umls = (
        reference_umls
        & comparison_umls
    )

    if len(paired_umls) != EXPECTED_PAIRED_UMLS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_PAIRED_UMLS} paired UMLs, "
            f"found {len(paired_umls)}."
        )

    if paired_umls != set(UML_LEVELS):
        raise ValueError(
            "Pilot contrast does not contain "
            "the expected ten UML models."
        )

    uml_order = UML_LEVELS

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
        reference["uml_model"].to_numpy(),
        comparison["uml_model"].to_numpy(),
    ):
        raise ValueError(
            "Reference and comparison UMLs "
            "are not aligned."
        )

    return (
        reference,
        comparison,
    )

def compute_weighted_estimands(
    dataframe,
    weights,
):
    numerator_mean = dataframe[
        "numerator_mean"
    ].to_numpy(dtype=float)

    denominator_mean = dataframe[
        "denominator_mean"
    ].to_numpy(dtype=float)

    cell_ratio = dataframe[
        "cell_ratio"
    ].to_numpy(dtype=float)

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

def compute_uniform_weight_diagnostics(
    reference,
    comparison,
):
    n_umls = len(reference)

    weights = np.full(
        n_umls,
        1.0 / n_umls,
        dtype=float,
    )

    (
        reference_element_weighted,
        reference_model_weighted,
    ) = compute_weighted_estimands(
        reference,
        weights,
    )

    (
        comparison_element_weighted,
        comparison_model_weighted,
    ) = compute_weighted_estimands(
        comparison,
        weights,
    )

    element_weighted_difference = (
        comparison_element_weighted
        - reference_element_weighted
    )

    model_weighted_difference = (
        comparison_model_weighted
        - reference_model_weighted
    )

    direct_reference_element_weighted = (
        reference["numerator_mean"].sum()
        / reference["denominator_mean"].sum()
    )

    direct_comparison_element_weighted = (
        comparison["numerator_mean"].sum()
        / comparison["denominator_mean"].sum()
    )

    direct_element_weighted_difference = (
        direct_comparison_element_weighted
        - direct_reference_element_weighted
    )

    direct_model_weighted_difference = (
        comparison["cell_ratio"].mean()
        - reference["cell_ratio"].mean()
    )

    if not np.isclose(
        element_weighted_difference,
        direct_element_weighted_difference,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Uniform-weight element-weighted "
            "contrast failed validation."
        )

    if not np.isclose(
        model_weighted_difference,
        direct_model_weighted_difference,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Uniform-weight model-weighted "
            "contrast failed validation."
        )

    return {
        "reference_element_weighted": float(
            reference_element_weighted
        ),
        "comparison_element_weighted": float(
            comparison_element_weighted
        ),
        "element_weighted_difference": float(
            element_weighted_difference
        ),
        "reference_model_weighted": float(
            reference_model_weighted
        ),
        "comparison_model_weighted": float(
            comparison_model_weighted
        ),
        "model_weighted_difference": float(
            model_weighted_difference
        ),
    }

def generate_bayesian_bootstrap_weights(
    n_units,
):
    if n_units < 2:
        raise ValueError(
            "Bayesian bootstrap requires "
            "at least two UML units for "
            "inferential use."
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
            "Unexpected Bayesian-bootstrap "
            f"weight shape: {weights.shape}."
        )

    if not np.allclose(
        weights.sum(axis=1),
        1.0,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Bayesian-bootstrap weights "
            "do not sum to one."
        )

    if (
        weights <= 0
    ).any():
        raise ValueError(
            "Bayesian-bootstrap weights "
            "must be strictly positive."
        )

    return weights

def compute_bayesian_bootstrap_contrasts(
    reference,
    comparison,
    weights,
):
    (
        reference_element_weighted,
        reference_model_weighted,
    ) = compute_weighted_estimands(
        reference,
        weights,
    )

    (
        comparison_element_weighted,
        comparison_model_weighted,
    ) = compute_weighted_estimands(
        comparison,
        weights,
    )

    element_weighted_difference = (
        comparison_element_weighted
        - reference_element_weighted
    )

    model_weighted_difference = (
        comparison_model_weighted
        - reference_model_weighted
    )

    if element_weighted_difference.shape != (
        BAYESIAN_BOOTSTRAP_DRAWS,
    ):
        raise ValueError(
            "Unexpected element-weighted "
            "contrast shape."
        )

    if model_weighted_difference.shape != (
        BAYESIAN_BOOTSTRAP_DRAWS,
    ):
        raise ValueError(
            "Unexpected model-weighted "
            "contrast shape."
        )

    if not np.isfinite(
        element_weighted_difference
    ).all():
        raise ValueError(
            "Element-weighted contrasts "
            "contain non-finite values."
        )

    if not np.isfinite(
        model_weighted_difference
    ).all():
        raise ValueError(
            "Model-weighted contrasts "
            "contain non-finite values."
        )

    return {
        "element_weighted": (
            element_weighted_difference
        ),
        "model_weighted": (
            model_weighted_difference
        ),
    }

def summarize_draws(
    draws,
):
    return {
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
            np.mean(
                draws > 0
            )
        ),
        "probability_lt_0": float(
            np.mean(
                draws < 0
            )
        ),
    }

def validate_direction_probabilities(
    summary,
):
    total = (
        summary["probability_gt_0"]
        + summary["probability_lt_0"]
    )

    if not np.isclose(
        total,
        1.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Posterior direction probabilities "
            "do not sum to one."
        )

def main():
    dataframe = load_pilot_data()

    validate_pilot_data(
        dataframe
    )

    (
        reference,
        comparison,
    ) = prepare_paired_contrast(
        dataframe
    )

    uniform_diagnostics = (
        compute_uniform_weight_diagnostics(
            reference,
            comparison,
        )
    )

    weights = (
        generate_bayesian_bootstrap_weights(
            len(reference)
        )
    )

    contrasts = (
        compute_bayesian_bootstrap_contrasts(
            reference,
            comparison,
            weights,
        )
    )

    element_summary = summarize_draws(
        contrasts["element_weighted"]
    )

    model_summary = summarize_draws(
        contrasts["model_weighted"]
    )

    validate_direction_probabilities(
        element_summary
    )

    validate_direction_probabilities(
        model_summary
    )

    print(
        "Cell-level Bayesian bootstrap "
        "pilot completed successfully."
    )

    print(
        f"outcome: {PILOT_OUTCOME}"
    )

    print(
        f"model: {PILOT_MODEL}"
    )

    print(
        "contrast: "
        f"{COMPARISON_REPRESENTATION} - "
        f"{REFERENCE_REPRESENTATION}"
    )

    print(
        f"paired_umls: {len(reference)}"
    )

    print()
    print(
        "Uniform-weight diagnostics:"
    )

    for key, value in (
        uniform_diagnostics.items()
    ):
        print(
            f"{key}: {value:.12f}"
        )

    print()
    print(
        "Bayesian bootstrap "
        "element-weighted:"
    )

    for key, value in (
        element_summary.items()
    ):
        print(
            f"{key}: {value:.12f}"
        )

    print()
    print(
        "Bayesian bootstrap "
        "model-weighted:"
    )

    for key, value in (
        model_summary.items()
    ):
        print(
            f"{key}: {value:.12f}"
        )


if __name__ == "__main__":
    main()