from pathlib import Path

import pandas as pd


CELL_LEVEL_DATA_PATH = Path(
    "experiment/analysis/data/cell_level"
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


def validate_cell_level_data(
    outcome_name,
    dataframe,
):
    required_columns = {
        "cell_id",
        "model",
        "uml_model",
        "representation",
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

    if not set(
        dataframe["model"]
    ).issubset(
        set(MODEL_LEVELS)
    ):
        raise ValueError(
            f"{outcome_name}: unexpected LLM."
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
            f"{outcome_name}: unexpected "
            "representation."
        )

    if not set(
        dataframe["uml_model"]
    ).issubset(
        set(UML_LEVELS)
    ):
        raise ValueError(
            f"{outcome_name}: unexpected UML."
        )

    if dataframe[
        "cell_ratio"
    ].isna().any():
        raise ValueError(
            f"{outcome_name}: cell_ratio "
            "contains NA."
        )


def sorted_uml_models(values):
    return sorted(
        values,
        key=lambda value: int(
            value.replace("uml", "")
        ),
    )


def build_contrast_support(
    outcome_name,
    dataframe,
):
    rows = []

    for model_name in MODEL_LEVELS:
        model_data = dataframe[
            dataframe["model"]
            == model_name
        ]

        reference_umls = set(
            model_data.loc[
                model_data["representation"]
                == REFERENCE_REPRESENTATION,
                "uml_model",
            ]
        )

        for representation_name in (
            COMPARISON_REPRESENTATIONS
        ):
            comparison_umls = set(
                model_data.loc[
                    model_data["representation"]
                    == representation_name,
                    "uml_model",
                ]
            )

            paired_umls = (
                reference_umls
                & comparison_umls
            )

            reference_only_umls = (
                reference_umls
                - comparison_umls
            )

            comparison_only_umls = (
                comparison_umls
                - reference_umls
            )

            neither_umls = (
                set(UML_LEVELS)
                - (
                    reference_umls
                    | comparison_umls
                )
            )

            rows.append(
                {
                    "outcome": outcome_name,
                    "model": model_name,
                    "reference": (
                        REFERENCE_REPRESENTATION
                    ),
                    "comparison": (
                        representation_name
                    ),
                    "n_reference_umls": len(
                        reference_umls
                    ),
                    "n_comparison_umls": len(
                        comparison_umls
                    ),
                    "n_paired_umls": len(
                        paired_umls
                    ),
                    "n_reference_only_umls": len(
                        reference_only_umls
                    ),
                    "n_comparison_only_umls": len(
                        comparison_only_umls
                    ),
                    "n_neither_umls": len(
                        neither_umls
                    ),
                    "paired_umls": ",".join(
                        sorted_uml_models(
                            paired_umls
                        )
                    ),
                    "reference_only_umls": (
                        ",".join(
                            sorted_uml_models(
                                reference_only_umls
                            )
                        )
                    ),
                    "comparison_only_umls": (
                        ",".join(
                            sorted_uml_models(
                                comparison_only_umls
                            )
                        )
                    ),
                    "neither_umls": ",".join(
                        sorted_uml_models(
                            neither_umls
                        )
                    ),
                }
            )

    result = pd.DataFrame(rows)

    expected_rows = (
        len(MODEL_LEVELS)
        * len(
            COMPARISON_REPRESENTATIONS
        )
    )

    if len(result) != expected_rows:
        raise ValueError(
            f"{outcome_name}: expected "
            f"{expected_rows} contrast-support "
            f"rows, found {len(result)}."
        )

    return result


def validate_contrast_support(
    outcome_name,
    support,
):
    if (
        support["n_paired_umls"]
        > 10
    ).any():
        raise ValueError(
            f"{outcome_name}: more than "
            "10 paired UML models."
        )

    count_columns = [
        "n_paired_umls",
        "n_reference_only_umls",
        "n_comparison_only_umls",
        "n_neither_umls",
    ]

    total = support[
        count_columns
    ].sum(axis=1)

    if not (
        total == len(UML_LEVELS)
    ).all():
        raise ValueError(
            f"{outcome_name}: UML support "
            "partition does not sum to 10."
        )

    reconstructed_reference = (
        support["n_paired_umls"]
        + support[
            "n_reference_only_umls"
        ]
    )

    if not (
        reconstructed_reference
        == support["n_reference_umls"]
    ).all():
        raise ValueError(
            f"{outcome_name}: invalid "
            "reference support counts."
        )

    reconstructed_comparison = (
        support["n_paired_umls"]
        + support[
            "n_comparison_only_umls"
        ]
    )

    if not (
        reconstructed_comparison
        == support["n_comparison_umls"]
    ).all():
        raise ValueError(
            f"{outcome_name}: invalid "
            "comparison support counts."
        )


def main():
    all_support = []

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

        support = build_contrast_support(
            outcome_name,
            dataframe,
        )

        validate_contrast_support(
            outcome_name,
            support,
        )

        all_support.append(
            support
        )

        print()
        print("=" * 80)
        print(outcome_name)
        print("=" * 80)

        print(
            support[
                [
                    "model",
                    "comparison",
                    "n_paired_umls",
                ]
            ].to_string(
                index=False
            )
        )

    combined = pd.concat(
        all_support,
        ignore_index=True,
    )

    output_filename = (
        CELL_LEVEL_DATA_PATH
        / "contrast_support.csv"
    )

    combined.to_csv(
        output_filename,
        index=False,
    )

    print()
    print(
        "Cell-level contrast support "
        "validated successfully."
    )

    print(
        f"rows: {len(combined)}"
    )

    print(
        f"saved: {output_filename}"
    )


if __name__ == "__main__":
    main()