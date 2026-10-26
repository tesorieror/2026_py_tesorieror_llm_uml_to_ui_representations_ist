from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_FILENAME = Path(
    "experiment/analysis/tables/"
    "primary_element_weighted_contrasts.csv"
)

SENSITIVITY_FILENAME = Path(
    "experiment/analysis/tables/"
    "sensitivity_model_weighted_contrasts.csv"
)

OUTPUT_PATH = Path(
    "experiment/analysis/tables"
)


EXPECTED_ROWS = 84

ZERO_TOLERANCE = 1e-15


KEY_COLUMNS = [
    "outcome",
    "model",
    "reference",
    "comparison",
    "n_paired_umls",
    "paired_umls",
    "analysis_status",
]

def load_inputs():
    if not PRIMARY_FILENAME.is_file():
        raise FileNotFoundError(
            f"Primary table not found: "
            f"{PRIMARY_FILENAME}"
        )

    if not SENSITIVITY_FILENAME.is_file():
        raise FileNotFoundError(
            f"Sensitivity table not found: "
            f"{SENSITIVITY_FILENAME}"
        )

    primary = pd.read_csv(
        PRIMARY_FILENAME
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_FILENAME
    )

    return primary, sensitivity

def validate_inputs(
    primary,
    sensitivity,
):
    primary_required = {
        *KEY_COLUMNS,
        "element_weighted_difference",
        "element_bb_mean",
        "element_bb_q025",
        "element_bb_q975",
        "element_bb_probability_gt_0",
        "element_bb_probability_lt_0",
        "element_bb_probability_eq_0",
    }

    sensitivity_required = {
        *KEY_COLUMNS,
        "model_weighted_difference",
        "model_weighted_bb_mean",
        "model_weighted_bb_q025",
        "model_weighted_bb_q975",
        "model_weighted_bb_probability_gt_0",
        "model_weighted_bb_probability_lt_0",
        "model_weighted_bb_probability_eq_0",
    }

    primary_missing = (
        primary_required
        - set(primary.columns)
    )

    if primary_missing:
        raise KeyError(
            "Primary table missing columns: "
            f"{sorted(primary_missing)}"
        )

    sensitivity_missing = (
        sensitivity_required
        - set(sensitivity.columns)
    )

    if sensitivity_missing:
        raise KeyError(
            "Sensitivity table missing columns: "
            f"{sorted(sensitivity_missing)}"
        )

    if len(primary) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} primary rows, "
            f"found {len(primary)}."
        )

    if len(sensitivity) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} sensitivity rows, "
            f"found {len(sensitivity)}."
        )

    if primary[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate primary rows."
        )

    if sensitivity[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate sensitivity rows."
        )

    primary_keys = (
        primary[
            KEY_COLUMNS
        ]
        .sort_values(
            KEY_COLUMNS[:4]
        )
        .reset_index(drop=True)
    )

    sensitivity_keys = (
        sensitivity[
            KEY_COLUMNS
        ]
        .sort_values(
            KEY_COLUMNS[:4]
        )
        .reset_index(drop=True)
    )

    if not primary_keys.equals(
        sensitivity_keys
    ):
        raise ValueError(
            "Primary and sensitivity tables "
            "do not contain identical contrast keys."
        )

def classify_pattern(
    status,
    lower,
    upper,
    probability_eq_0,
):
    if status == "not_estimable":
        return "not_estimable"

    if status == "descriptive_only":
        return "descriptive_only"

    if status != "bayesian_bootstrap":
        raise ValueError(
            f"Unexpected analysis status: "
            f"{status}"
        )

    if np.isclose(
        probability_eq_0,
        1.0,
        rtol=0.0,
        atol=ZERO_TOLERANCE,
    ):
        return "exact_zero"

    if lower > 0:
        return "above_zero"

    if upper < 0:
        return "below_zero"

    return "contains_zero"

def build_comparison(
    primary,
    sensitivity,
):
    primary_selected = primary[
        KEY_COLUMNS
        + [
            "element_weighted_difference",
            "element_bb_mean",
            "element_bb_q025",
            "element_bb_q975",
            "element_bb_probability_gt_0",
            "element_bb_probability_lt_0",
            "element_bb_probability_eq_0",
        ]
    ].copy()

    sensitivity_selected = sensitivity[
        KEY_COLUMNS
        + [
            "model_weighted_difference",
            "model_weighted_bb_mean",
            "model_weighted_bb_q025",
            "model_weighted_bb_q975",
            "model_weighted_bb_probability_gt_0",
            "model_weighted_bb_probability_lt_0",
            "model_weighted_bb_probability_eq_0",
        ]
    ].copy()

    result = primary_selected.merge(
        sensitivity_selected,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(result) != EXPECTED_ROWS:
        raise ValueError(
            "Primary-sensitivity merge "
            "did not preserve 84 contrasts."
        )

    result["primary_pattern"] = [
        classify_pattern(
            status,
            lower,
            upper,
            probability_zero,
        )
        for (
            status,
            lower,
            upper,
            probability_zero,
        ) in zip(
            result["analysis_status"],
            result["element_bb_q025"],
            result["element_bb_q975"],
            result[
                "element_bb_probability_eq_0"
            ],
        )
    ]

    result["sensitivity_pattern"] = [
        classify_pattern(
            status,
            lower,
            upper,
            probability_zero,
        )
        for (
            status,
            lower,
            upper,
            probability_zero,
        ) in zip(
            result["analysis_status"],
            result[
                "model_weighted_bb_q025"
            ],
            result[
                "model_weighted_bb_q975"
            ],
            result[
                "model_weighted_bb_probability_eq_0"
            ],
        )
    ]

    result["pattern_agreement"] = (
        result["primary_pattern"]
        == result["sensitivity_pattern"]
    )

    result["pattern_transition"] = (
        result["primary_pattern"]
        + " -> "
        + result["sensitivity_pattern"]
    )

    return result

def add_mean_sign_diagnostics(
    dataframe,
):
    result = dataframe.copy()

    bootstrap_mask = (
        result["analysis_status"]
        == "bayesian_bootstrap"
    )

    result[
        "posterior_mean_sign_flip"
    ] = False

    primary_mean = result[
        "element_bb_mean"
    ]

    sensitivity_mean = result[
        "model_weighted_bb_mean"
    ]

    result.loc[
        bootstrap_mask,
        "posterior_mean_sign_flip",
    ] = (
        (
            primary_mean[
                bootstrap_mask
            ]
            * sensitivity_mean[
                bootstrap_mask
            ]
        )
        < 0
    )

    return result

def validate_comparison(
    dataframe,
):
    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            "Expected 84 comparison rows."
        )

    if dataframe[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate comparison rows."
        )

    allowed_patterns = {
        "above_zero",
        "below_zero",
        "exact_zero",
        "contains_zero",
        "descriptive_only",
        "not_estimable",
    }

    if not set(
        dataframe["primary_pattern"]
    ).issubset(
        allowed_patterns
    ):
        raise ValueError(
            "Unexpected primary pattern."
        )

    if not set(
        dataframe["sensitivity_pattern"]
    ).issubset(
        allowed_patterns
    ):
        raise ValueError(
            "Unexpected sensitivity pattern."
        )

    non_bootstrap = (
        dataframe[
            "analysis_status"
        ] != "bayesian_bootstrap"
    )

    if dataframe.loc[
        non_bootstrap,
        "posterior_mean_sign_flip",
    ].any():
        raise ValueError(
            "Non-bootstrap contrast marked "
            "as posterior mean sign flip."
        )

def summarize_transitions(
    dataframe,
):
    return (
        dataframe[
            "pattern_transition"
        ]
        .value_counts()
        .rename_axis(
            "pattern_transition"
        )
        .reset_index(
            name="n_contrasts"
        )
    )


def summarize_agreement(
    dataframe,
):
    total = len(dataframe)

    n_agree = int(
        dataframe[
            "pattern_agreement"
        ].sum()
    )

    n_disagree = (
        total - n_agree
    )

    n_sign_flip = int(
        dataframe[
            "posterior_mean_sign_flip"
        ].sum()
    )

    return pd.DataFrame(
        [
            {
                "n_total": total,
                "n_pattern_agreement": (
                    n_agree
                ),
                "n_pattern_disagreement": (
                    n_disagree
                ),
                "pattern_agreement_rate": (
                    n_agree / total
                ),
                "n_posterior_mean_sign_flip": (
                    n_sign_flip
                ),
            }
        ]
    )    

def write_table(
    dataframe,
    filename,
):
    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_filename = (
        OUTPUT_PATH
        / filename
    )

    dataframe.to_csv(
        output_filename,
        index=False,
    )

    return output_filename

def main():
    primary, sensitivity = (
        load_inputs()
    )

    validate_inputs(
        primary,
        sensitivity,
    )

    comparison = build_comparison(
        primary,
        sensitivity,
    )

    comparison = (
        add_mean_sign_diagnostics(
            comparison
        )
    )

    validate_comparison(
        comparison
    )

    transitions = (
        summarize_transitions(
            comparison
        )
    )

    agreement = (
        summarize_agreement(
            comparison
        )
    )

    comparison_filename = (
        write_table(
            comparison,
            "primary_sensitivity_comparison.csv",
        )
    )

    transitions_filename = (
        write_table(
            transitions,
            "primary_sensitivity_transitions.csv",
        )
    )

    agreement_filename = (
        write_table(
            agreement,
            "primary_sensitivity_agreement.csv",
        )
    )

    print(
        "Primary vs sensitivity comparison "
        "completed and validated successfully."
    )

    print()
    print(
        "Agreement summary:"
    )

    print(
        agreement.to_string(
            index=False
        )
    )

    print()
    print(
        "Pattern transitions:"
    )

    print(
        transitions.to_string(
            index=False
        )
    )

    print()
    print(
        "Contrasts with different patterns:"
    )

    print(
        comparison.loc[
            ~comparison[
                "pattern_agreement"
            ],
            [
                "outcome",
                "model",
                "comparison",
                "n_paired_umls",
                "primary_pattern",
                "sensitivity_pattern",
                "element_bb_mean",
                "model_weighted_bb_mean",
            ],
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Posterior mean sign flips:"
    )

    print(
        comparison.loc[
            comparison[
                "posterior_mean_sign_flip"
            ],
            [
                "outcome",
                "model",
                "comparison",
                "n_paired_umls",
                "primary_pattern",
                "sensitivity_pattern",
                "element_bb_mean",
                "model_weighted_bb_mean",
            ],
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"saved: {comparison_filename}"
    )

    print(
        f"saved: {transitions_filename}"
    )

    print(
        f"saved: {agreement_filename}"
    )


if __name__ == "__main__":
    main()