from pathlib import Path

import pandas as pd


PRIMARY_RESULTS_PATH = Path(
    "experiment/analysis/results/"
    "cell_level_bayesian_bootstrap/"
    "primary_contrasts.csv"
)

PRIMARY_TABLE_PATH = Path(
    "experiment/analysis/tables/"
    "primary_element_weighted_contrasts.csv"
)

SENSITIVITY_TABLE_PATH = Path(
    "experiment/analysis/tables/"
    "sensitivity_model_weighted_contrasts.csv"
)

PRIMARY_RESULTS_TABLE_PATH = Path(
    "experiment/analysis/tables/"
    "primary_results_table.csv"
)

PRIMARY_SUPPORT_TABLE_PATH = Path(
    "experiment/analysis/tables/"
    "primary_results_support.csv"
)

LOO_SUMMARY_PATH = Path(
    "experiment/analysis/tables/"
    "leave_one_uml_out_summary.csv"
)

GENERATION_MARGINAL_PATH = Path(
    "experiment/analysis/results/"
    "generation_success_beta_binomial/"
    "marginal_probabilities.csv"
)

GENERATION_CONTRAST_PATH = Path(
    "experiment/analysis/results/"
    "generation_success_beta_binomial/"
    "primary_contrasts.csv"
)

GENERATION_RESULTS_TABLE_PATH = Path(
    "experiment/analysis/tables/"
    "generation_success_results_table.csv"
)

SECONDARY_EVALUABILITY_PATH = Path(
    "experiment/analysis/results/secondary/"
    "association_cardinality_evaluability/"
    "contrasts.csv"
)

PRIMARY_FIGURE_PNG = Path(
    "experiment/analysis/figures/"
    "primary_element_weighted_contrasts.png"
)

PRIMARY_FIGURE_PDF = Path(
    "experiment/analysis/figures/"
    "primary_element_weighted_contrasts.pdf"
)

CSV_ARTIFACTS = {
    "primary_contrasts": (
        PRIMARY_RESULTS_PATH,
        84,
    ),
    "primary_element_weighted": (
        PRIMARY_TABLE_PATH,
        84,
    ),
    "model_weighted_sensitivity": (
        SENSITIVITY_TABLE_PATH,
        84,
    ),
    "primary_results_table": (
        PRIMARY_RESULTS_TABLE_PATH,
        21,
    ),
    "primary_results_support": (
        PRIMARY_SUPPORT_TABLE_PATH,
        21,
    ),
    "leave_one_uml_out_summary": (
        LOO_SUMMARY_PATH,
        76,
    ),
    "generation_marginal_probabilities": (
        GENERATION_MARGINAL_PATH,
        15,
    ),
    "generation_primary_contrasts": (
        GENERATION_CONTRAST_PATH,
        12,
    ),
    "generation_results_table": (
        GENERATION_RESULTS_TABLE_PATH,
        3,
    ),
    "secondary_association_cardinality_evaluability": (
        SECONDARY_EVALUABILITY_PATH,
        12,
    ),
}

def validate_csv_artifacts():
    rows = []

    for name, (
        filename,
        expected_rows,
    ) in CSV_ARTIFACTS.items():
        if not filename.is_file():
            raise FileNotFoundError(
                f"Final analysis artifact "
                f"not found: {filename}"
            )

        dataframe = pd.read_csv(
            filename
        )

        actual_rows = len(
            dataframe
        )

        if actual_rows != expected_rows:
            raise ValueError(
                f"{name}: expected "
                f"{expected_rows} rows, "
                f"found {actual_rows}."
            )

        rows.append(
            {
                "artifact": name,
                "path": str(filename),
                "rows": actual_rows,
                "columns": len(
                    dataframe.columns
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def validate_primary_results():
    primary = pd.read_csv(
        PRIMARY_RESULTS_PATH
    )

    if primary[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate primary contrasts."
        )

    expected_status = {
        "bayesian_bootstrap": 76,
        "descriptive_only": 7,
        "not_estimable": 1,
    }

    observed_status = (
        primary[
            "analysis_status"
        ]
        .value_counts()
        .to_dict()
    )

    if observed_status != expected_status:
        raise ValueError(
            "Unexpected primary analysis-status "
            f"distribution: {observed_status}"
        )

def validate_primary_sensitivity_alignment():
    primary = pd.read_csv(
        PRIMARY_TABLE_PATH
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_TABLE_PATH
    )

    key_columns = [
        "outcome",
        "model",
        "reference",
        "comparison",
        "n_paired_umls",
        "paired_umls",
        "analysis_status",
    ]

    primary_keys = (
        primary[
            key_columns
        ]
        .sort_values(
            key_columns[:4]
        )
        .reset_index(
            drop=True
        )
    )

    sensitivity_keys = (
        sensitivity[
            key_columns
        ]
        .sort_values(
            key_columns[:4]
        )
        .reset_index(
            drop=True
        )
    )

    if not primary_keys.equals(
        sensitivity_keys
    ):
        raise ValueError(
            "Primary and model-weighted "
            "sensitivity results are no "
            "longer aligned."
        )

def validate_generation_success():
    marginal = pd.read_csv(
        GENERATION_MARGINAL_PATH
    )

    contrasts = pd.read_csv(
        GENERATION_CONTRAST_PATH
    )

    if int(
        marginal[
            "observed_successes"
        ].sum()
    ) != 1479:
        raise ValueError(
            "Generation-success outputs "
            "do not preserve 1479 successes."
        )

    if int(
        marginal[
            "observed_trials"
        ].sum()
    ) != 1500:
        raise ValueError(
            "Generation-success outputs "
            "do not preserve 1500 trials."
        )

    if contrasts[
        [
            "model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate generation-success "
            "contrasts."
        )

def validate_secondary_evaluability():
    secondary = pd.read_csv(
        SECONDARY_EVALUABILITY_PATH
    )

    expected_status = {
        "bayesian_bootstrap": 8,
        "descriptive_only": 3,
        "not_estimable": 1,
    }

    observed_status = (
        secondary[
            "analysis_status"
        ]
        .value_counts()
        .to_dict()
    )

    if observed_status != expected_status:
        raise ValueError(
            "Unexpected secondary evaluability "
            f"status distribution: "
            f"{observed_status}"
        )

def validate_figures():
    for filename in [
        PRIMARY_FIGURE_PNG,
        PRIMARY_FIGURE_PDF,
    ]:
        if not filename.is_file():
            raise FileNotFoundError(
                f"Final figure not found: "
                f"{filename}"
            )

        if filename.stat().st_size <= 0:
            raise ValueError(
                f"Final figure is empty: "
                f"{filename}"
            )

INVENTORY_FILENAME = Path(
    "experiment/analysis/"
    "final_analysis_inventory.csv"
)

def write_inventory(
    inventory,
):
    INVENTORY_FILENAME.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        INVENTORY_FILENAME,
        index=False,
    )





def main():
    inventory = (
        validate_csv_artifacts()
    )

    validate_primary_results()

    validate_primary_sensitivity_alignment()

    validate_generation_success()

    validate_secondary_evaluability()

    validate_figures()

    write_inventory(
        inventory
    )

    print(
        "Final statistical-analysis outputs "
        "validated successfully."
    )

    print()

    print(
        inventory.to_string(
            index=False
        )
    )

    print()

    print(
        "Primary coverage/fidelity:"
    )

    print(
        "  84 contrasts "
        "(76 inferential, "
        "7 descriptive, "
        "1 not estimable)"
    )

    print(
        "Generation success:"
    )

    print(
        "  15 marginal probabilities, "
        "12 primary contrasts"
    )

    print(
        "Secondary association-cardinality "
        "evaluability:"
    )

    print(
        "  12 contrasts "
        "(8 inferential, "
        "3 descriptive, "
        "1 not estimable)"
    )

    print(
        "Leave-one-UML-out:"
    )

    print(
        "  76 inferential contrasts audited"
    )

    print()

    print(
        f"saved: {INVENTORY_FILENAME}"
    )
    

if __name__ == "__main__":
    main()



