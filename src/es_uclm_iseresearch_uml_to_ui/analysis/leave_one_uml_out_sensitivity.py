from pathlib import Path

import numpy as np
import pandas as pd


CELL_LEVEL_DATA_PATH = Path(
    "experiment/analysis/data/cell_level"
)

PRIMARY_FILENAME = Path(
    "experiment/analysis/tables/"
    "primary_element_weighted_contrasts.csv"
)

OUTPUT_PATH = Path(
    "experiment/analysis/tables"
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


COMPARISON_LEVELS = [
    "json",
    "xsd",
    "sql",
    "txt",
]


REFERENCE_REPRESENTATION = "uml"

EXPECTED_PRIMARY_ROWS = 84

def load_primary_contrasts():
    if not PRIMARY_FILENAME.is_file():
        raise FileNotFoundError(
            f"Primary contrasts not found: "
            f"{PRIMARY_FILENAME}"
        )

    return pd.read_csv(
        PRIMARY_FILENAME
    )


def load_cell_level_data(
    outcome_name,
):
    filename = (
        CELL_LEVEL_DATA_PATH
        / f"{outcome_name}.csv"
    )

    if not filename.is_file():
        raise FileNotFoundError(
            f"Cell-level dataset not found: "
            f"{filename}"
        )

    return pd.read_csv(
        filename
    )

def validate_primary_contrasts(
    dataframe,
):
    required_columns = {
        "outcome",
        "model",
        "reference",
        "comparison",
        "n_paired_umls",
        "paired_umls",
        "analysis_status",
        "element_weighted_difference",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Primary contrast table missing: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != EXPECTED_PRIMARY_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_PRIMARY_ROWS} "
            f"primary rows, found {len(dataframe)}."
        )

    if dataframe[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate primary contrast rows."
        )

    if not (
        dataframe["reference"]
        == REFERENCE_REPRESENTATION
    ).all():
        raise ValueError(
            "Unexpected reference representation."
        )

def compute_element_weighted_ratio(
    dataframe,
):
    denominator = float(
        dataframe[
            "denominator_mean"
        ].sum()
    )

    if denominator <= 0:
        raise ValueError(
            "Non-positive aggregate denominator."
        )

    numerator = float(
        dataframe[
            "numerator_mean"
        ].sum()
    )

    return numerator / denominator

def compute_contrast(
    reference,
    comparison,
):
    if len(reference) != len(comparison):
        raise ValueError(
            "Reference and comparison sizes differ."
        )

    if len(reference) == 0:
        raise ValueError(
            "Cannot calculate empty contrast."
        )

    reference_ratio = (
        compute_element_weighted_ratio(
            reference
        )
    )

    comparison_ratio = (
        compute_element_weighted_ratio(
            comparison
        )
    )

    return (
        comparison_ratio
        - reference_ratio
    )


def build_loo_rows(
    outcome_name,
    cell_data,
    primary,
):
    rows = []

    outcome_primary = primary[
        primary["outcome"]
        == outcome_name
    ]

    for _, primary_row in (
        outcome_primary.iterrows()
    ):
        if (
            primary_row[
                "analysis_status"
            ]
            != "bayesian_bootstrap"
        ):
            continue

        model_name = primary_row[
            "model"
        ]

        comparison_name = primary_row[
            "comparison"
        ]

        paired_umls = [
            value
            for value in str(
                primary_row[
                    "paired_umls"
                ]
            ).split(",")
            if value
        ]

        n_paired = len(
            paired_umls
        )

        if n_paired < 2:
            raise ValueError(
                "Bootstrap contrast has fewer "
                "than two paired UMLs."
            )

        model_data = cell_data[
            cell_data["model"]
            == model_name
        ]

        reference = (
            model_data[
                (
                    model_data[
                        "representation"
                    ]
                    == REFERENCE_REPRESENTATION
                )
                & (
                    model_data[
                        "uml_model"
                    ].isin(
                        paired_umls
                    )
                )
            ]
            .copy()
        )

        comparison = (
            model_data[
                (
                    model_data[
                        "representation"
                    ]
                    == comparison_name
                )
                & (
                    model_data[
                        "uml_model"
                    ].isin(
                        paired_umls
                    )
                )
            ]
            .copy()
        )

        if (
            set(reference["uml_model"])
            != set(paired_umls)
        ):
            raise ValueError(
                "Reference UML support mismatch."
            )

        if (
            set(comparison["uml_model"])
            != set(paired_umls)
        ):
            raise ValueError(
                "Comparison UML support mismatch."
            )

        full_difference = (
            compute_contrast(
                reference,
                comparison,
            )
        )

        stored_difference = float(
            primary_row[
                "element_weighted_difference"
            ]
        )

        if not np.isclose(
            full_difference,
            stored_difference,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "Reconstructed full contrast "
                "does not match primary result "
                f"for {outcome_name} / "
                f"{model_name} / "
                f"{comparison_name}."
            )

        for omitted_uml in paired_umls:
            retained_umls = [
                uml
                for uml in paired_umls
                if uml != omitted_uml
            ]

            reference_loo = reference[
                reference[
                    "uml_model"
                ].isin(
                    retained_umls
                )
            ]

            comparison_loo = comparison[
                comparison[
                    "uml_model"
                ].isin(
                    retained_umls
                )
            ]

            loo_difference = (
                compute_contrast(
                    reference_loo,
                    comparison_loo,
                )
            )

            rows.append(
                {
                    "outcome": outcome_name,
                    "model": model_name,
                    "comparison": comparison_name,
                    "n_paired_umls": n_paired,
                    "omitted_uml": omitted_uml,
                    "n_retained_umls": (
                        len(retained_umls)
                    ),
                    "full_difference": (
                        full_difference
                    ),
                    "loo_difference": (
                        loo_difference
                    ),
                    "absolute_change": abs(
                        loo_difference
                        - full_difference
                    ),
                }
            )

    return rows

def numeric_sign(
    value,
    tolerance=1e-15,
):
    if value > tolerance:
        return 1

    if value < -tolerance:
        return -1

    return 0

def add_sign_diagnostics(
    dataframe,
):
    result = dataframe.copy()

    result["full_sign"] = (
        result[
            "full_difference"
        ].apply(
            numeric_sign
        )
    )

    result["loo_sign"] = (
        result[
            "loo_difference"
        ].apply(
            numeric_sign
        )
    )

    result["sign_change"] = (
        result["full_sign"]
        != result["loo_sign"]
    )

    return result

def summarize_loo(
    dataframe,
):
    summary = (
        dataframe
        .groupby(
            [
                "outcome",
                "model",
                "comparison",
                "n_paired_umls",
            ],
            as_index=False,
        )
        .agg(
            full_difference=(
                "full_difference",
                "first",
            ),
            loo_min=(
                "loo_difference",
                "min",
            ),
            loo_max=(
                "loo_difference",
                "max",
            ),
            max_absolute_change=(
                "absolute_change",
                "max",
            ),
            n_sign_changes=(
                "sign_change",
                "sum",
            ),
        )
    )

    return summary

def validate_loo_results(
    detailed,
    summary,
    primary,
):
    expected_contrasts = int(
        (
            primary[
                "analysis_status"
            ]
            == "bayesian_bootstrap"
        ).sum()
    )

    if len(summary) != expected_contrasts:
        raise ValueError(
            "LOO summary does not cover all "
            "Bayesian-bootstrap contrasts."
        )

    if expected_contrasts != 76:
        raise ValueError(
            "Expected 76 inferential contrasts."
        )

    expected_detail_rows = int(
        primary.loc[
            primary[
                "analysis_status"
            ]
            == "bayesian_bootstrap",
            "n_paired_umls",
        ].sum()
    )

    if len(detailed) != expected_detail_rows:
        raise ValueError(
            "Unexpected number of detailed "
            "leave-one-UML-out rows."
        )

    if not (
        detailed["n_retained_umls"]
        == detailed["n_paired_umls"] - 1
    ).all():
        raise ValueError(
            "Invalid retained UML count."
        )

    if (
        detailed["absolute_change"] < 0
    ).any():
        raise ValueError(
            "Negative absolute change."
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
    primary = (
        load_primary_contrasts()
    )

    validate_primary_contrasts(
        primary
    )

    all_rows = []

    for outcome_name in OUTCOMES:
        cell_data = (
            load_cell_level_data(
                outcome_name
            )
        )

        all_rows.extend(
            build_loo_rows(
                outcome_name,
                cell_data,
                primary,
            )
        )

    detailed = pd.DataFrame(
        all_rows
    )

    detailed = add_sign_diagnostics(
        detailed
    )

    summary = summarize_loo(
        detailed
    )

    validate_loo_results(
        detailed,
        summary,
        primary,
    )

    detailed_filename = write_table(
        detailed,
        "leave_one_uml_out_detailed.csv",
    )

    summary_filename = write_table(
        summary,
        "leave_one_uml_out_summary.csv",
    )

    print(
        "Leave-one-UML-out sensitivity "
        "completed and validated successfully."
    )

    print(
        f"inferential_contrasts: "
        f"{len(summary)}"
    )

    print(
        f"loo_rows: {len(detailed)}"
    )

    print(
        "contrasts_with_any_sign_change: "
        f"{int((summary['n_sign_changes'] > 0).sum())}"
    )

    print()

    print(
        "Largest LOO changes:"
    )

    print(
        summary
        .sort_values(
            "max_absolute_change",
            ascending=False,
        )
        .head(15)
        [
            [
                "outcome",
                "model",
                "comparison",
                "n_paired_umls",
                "full_difference",
                "loo_min",
                "loo_max",
                "max_absolute_change",
                "n_sign_changes",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"saved: {detailed_filename}"
    )

    print(
        f"saved: {summary_filename}"
    )


if __name__ == "__main__":
    main()

    

