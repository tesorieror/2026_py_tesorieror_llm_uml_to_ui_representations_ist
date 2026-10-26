from pathlib import Path

import numpy as np
import pandas as pd


CELL_LEVEL_DATA_PATH = Path(
    "experiment/analysis/data/cell_level"
)

SUPPORT_FILENAME = (
    CELL_LEVEL_DATA_PATH
    / "contrast_support.csv"
)

RESULTS_PATH = Path(
    "experiment/analysis/results/"
    "cell_level_bayesian_bootstrap"
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

def load_cell_level_data(
    outcome_name,
):
    if outcome_name not in OUTCOMES:
        raise KeyError(
            f"Unknown outcome: {outcome_name}"
        )

    filename = (
        CELL_LEVEL_DATA_PATH
        / f"{outcome_name}.csv"
    )

    if not filename.is_file():
        raise FileNotFoundError(
            f"Cell-level dataset not found: "
            f"{filename}"
        )

    return pd.read_csv(filename)


def load_contrast_support():
    if not SUPPORT_FILENAME.is_file():
        raise FileNotFoundError(
            "Contrast-support dataset not found: "
            f"{SUPPORT_FILENAME}"
        )

    return pd.read_csv(
        SUPPORT_FILENAME
    )

def validate_cell_level_data(
    outcome_name,
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
            f"{outcome_name}: missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe[
        [
            "model",
            "uml_model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            f"{outcome_name}: duplicate "
            "LLM × UML × representation cells."
        )

    if (
        dataframe["denominator_sum"] <= 0
    ).any():
        raise ValueError(
            f"{outcome_name}: denominator_sum "
            "must be positive."
        )

    if (
        dataframe["denominator_mean"] <= 0
    ).any():
        raise ValueError(
            f"{outcome_name}: denominator_mean "
            "must be positive."
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
            f"{outcome_name}: cell_ratio "
            "does not match aggregated counts."
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
            f"{outcome_name}: cell_ratio "
            "does not match numerator_mean / "
            "denominator_mean."
        )

def parse_uml_list(
    value,
):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return value.split(",")


def sort_uml_models(
    values,
):
    return sorted(
        values,
        key=lambda value: int(
            value.replace("uml", "")
        ),
    )


def validate_contrast_support(
    support,
):
    required_columns = {
        "outcome",
        "model",
        "reference",
        "comparison",
        "n_paired_umls",
        "paired_umls",
    }

    missing_columns = (
        required_columns
        - set(support.columns)
    )

    if missing_columns:
        raise KeyError(
            "contrast_support.csv is missing "
            f"columns: {sorted(missing_columns)}"
        )

    expected_rows = (
        len(OUTCOMES)
        * len(MODEL_LEVELS)
        * len(COMPARISON_REPRESENTATIONS)
    )

    if len(support) != expected_rows:
        raise ValueError(
            "Expected "
            f"{expected_rows} contrast-support "
            f"rows, found {len(support)}."
        )

    if support[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate contrast-support rows."
        )

    if not (
        support["reference"]
        == REFERENCE_REPRESENTATION
    ).all():
        raise ValueError(
            "Unexpected reference representation "
            "in contrast_support.csv."
        )

    for _, row in support.iterrows():
        paired_umls = parse_uml_list(
            row["paired_umls"]
        )

        if len(paired_umls) != int(
            row["n_paired_umls"]
        ):
            raise ValueError(
                "n_paired_umls does not match "
                "paired_umls for "
                f"{row['outcome']} / "
                f"{row['model']} / "
                f"{row['comparison']}."
            )

        if not set(
            paired_umls
        ).issubset(
            set(UML_LEVELS)
        ):
            raise ValueError(
                "Unexpected UML in "
                "contrast-support data."
            )

def prepare_paired_contrast(
    outcome_name,
    model_name,
    comparison_name,
    dataframe,
    support,
):
    support_rows = support[
        (
            support["outcome"]
            == outcome_name
        )
        & (
            support["model"]
            == model_name
        )
        & (
            support["comparison"]
            == comparison_name
        )
    ]

    if len(support_rows) != 1:
        raise ValueError(
            "Expected exactly one support row "
            f"for {outcome_name} / "
            f"{model_name} / "
            f"{comparison_name}."
        )

    support_row = support_rows.iloc[0]

    expected_paired_umls = set(
        parse_uml_list(
            support_row["paired_umls"]
        )
    )

    n_paired_umls = int(
        support_row["n_paired_umls"]
    )

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

    actual_paired_umls = (
        set(reference["uml_model"])
        & set(comparison["uml_model"])
    )

    if (
        actual_paired_umls
        != expected_paired_umls
    ):
        raise ValueError(
            "Cell-level data and "
            "contrast-support data disagree "
            f"for {outcome_name} / "
            f"{model_name} / "
            f"{comparison_name}."
        )

    if (
        len(actual_paired_umls)
        != n_paired_umls
    ):
        raise ValueError(
            "Invalid paired UML count."
        )

    uml_order = sort_uml_models(
        actual_paired_umls
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
                actual_paired_umls
            )
        ]
        .set_index("uml_model")
        .loc[uml_order]
        .reset_index()
    )

    comparison = (
        comparison[
            comparison["uml_model"].isin(
                actual_paired_umls
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
        uml_order,
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

    if n_umls == 0:
        raise ValueError(
            "Uniform-weight diagnostics require "
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
        reference["numerator_mean"].sum()
        / reference["denominator_mean"].sum()
    )

    direct_comparison_element = (
        comparison["numerator_mean"].sum()
        / comparison["denominator_mean"].sum()
    )

    direct_element_difference = (
        direct_comparison_element
        - direct_reference_element
    )

    direct_model_difference = (
        comparison["cell_ratio"].mean()
        - reference["cell_ratio"].mean()
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
            "Bootstrap weights do not sum "
            "to one."
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

    summary = {
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
        summary["probability_gt_0"]
        + summary["probability_lt_0"]
        + summary["probability_eq_0"]
    )

    if not np.isclose(
        probability_sum,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Direction probabilities do not "
            "sum to one."
        )

    return summary

def analyze_contrast(
    outcome_name,
    model_name,
    comparison_name,
    dataframe,
    support,
):
    (
        reference,
        comparison,
        paired_umls,
    ) = prepare_paired_contrast(
        outcome_name,
        model_name,
        comparison_name,
        dataframe,
        support,
    )

    n_paired = len(
        paired_umls
    )

    result = {
        "outcome": outcome_name,
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
        result["analysis_status"] = (
            "not_estimable"
        )

        diagnostic_columns = [
            "reference_element_weighted",
            "comparison_element_weighted",
            "element_weighted_difference",
            "reference_model_weighted",
            "comparison_model_weighted",
            "model_weighted_difference",
        ]

        for column in diagnostic_columns:
            result[column] = np.nan




        for prefix in [
            "element",
            "model_weighted",
        ]:
            for suffix in [
                "bb_mean",
                "bb_median",
                "bb_q025",
                "bb_q975",
                "bb_probability_gt_0",
                "bb_probability_lt_0",
                "bb_probability_eq_0",
            ]:
                result[
                    f"{prefix}_{suffix}"
                ] = np.nan


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
        result["analysis_status"] = (
            "descriptive_only"
        )

        for prefix in [
            "element",
            "model_weighted",
        ]:
            for suffix in [
                "bb_mean",
                "bb_median",
                "bb_q025",
                "bb_q975",
                "bb_probability_gt_0",
                "bb_probability_lt_0",
                "bb_probability_eq_0",
            ]:
                result[
                    f"{prefix}_{suffix}"
                ] = np.nan

        add_empty_bootstrap_summary(
            result
        )

        return result


    result["analysis_status"] = (
        "bayesian_bootstrap"
    )

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

def validate_results(
    results,
):
    expected_rows = (
        len(OUTCOMES)
        * len(MODEL_LEVELS)
        * len(COMPARISON_REPRESENTATIONS)
    )

    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} results, "
            f"found {len(results)}."
        )

    if results[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate contrast results."
        )

    allowed_status = {
        "not_estimable",
        "descriptive_only",
        "bayesian_bootstrap",
    }

    if not set(
        results["analysis_status"]
    ).issubset(
        allowed_status
    ):
        raise ValueError(
            "Unexpected analysis status."
        )

    if (
        results.loc[
            results["n_paired_umls"] == 0,
            "analysis_status",
        ]
        != "not_estimable"
    ).any():
        raise ValueError(
            "n=0 contrast has invalid status."
        )

    if (
        results.loc[
            results["n_paired_umls"] == 1,
            "analysis_status",
        ]
        != "descriptive_only"
    ).any():
        raise ValueError(
            "n=1 contrast has invalid status."
        )

    if (
        results.loc[
            results["n_paired_umls"] >= 2,
            "analysis_status",
        ]
        != "bayesian_bootstrap"
    ).any():
        raise ValueError(
            "Inferential contrast has invalid "
            "analysis status."
        )

def main():
    support = load_contrast_support()

    validate_contrast_support(
        support
    )

    rows = []

    for outcome_name in OUTCOMES:
        dataframe = (
            load_cell_level_data(
                outcome_name
            )
        )

        validate_cell_level_data(
            outcome_name,
            dataframe,
        )

        for model_name in MODEL_LEVELS:
            for comparison_name in (
                COMPARISON_REPRESENTATIONS
            ):
                result = analyze_contrast(
                    outcome_name,
                    model_name,
                    comparison_name,
                    dataframe,
                    support,
                )

                rows.append(
                    result
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

    filename = (
        RESULTS_PATH
        / "primary_contrasts.csv"
    )

    results.to_csv(
        filename,
        index=False,
    )

    print(
        "Cell-level Bayesian bootstrap "
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
        "support_distribution:"
    )

    print(
        results[
            "n_paired_umls"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Pilot validation:"
    )

    pilot = results[
        (
            results["outcome"]
            == "attribute_coverage"
        )
        & (
            results["model"]
            == "llama3.1:8b"
        )
        & (
            results["comparison"]
            == "json"
        )
    ]

    print(
        pilot.to_string(
            index=False
        )
    )

    print()
    print(
        f"saved: {filename}"
    )


if __name__ == "__main__":
    main()