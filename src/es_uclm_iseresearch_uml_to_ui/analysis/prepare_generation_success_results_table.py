from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_PATH = Path(
    "experiment/analysis/results/"
    "generation_success_beta_binomial"
)

MARGINAL_FILENAME = (
    RESULTS_PATH
    / "marginal_probabilities.csv"
)

CONTRAST_FILENAME = (
    RESULTS_PATH
    / "primary_contrasts.csv"
)

OUTPUT_PATH = Path(
    "experiment/analysis/tables"
)


MODEL_LEVELS = [
    "llama3.1:8b",
    "mistral:7b-instruct",
    "qwen2.5:7b-instruct",
]


REPRESENTATION_LEVELS = [
    "uml",
    "json",
    "xsd",
    "sql",
    "txt",
]


REPRESENTATION_LABELS = {
    "uml": "UML/XMI",
    "json": "JSON",
    "xsd": "XSD",
    "sql": "SQL",
    "txt": "CNL",
}


REFERENCE_REPRESENTATION = "uml"

EXPECTED_MARGINAL_ROWS = 15
EXPECTED_CONTRAST_ROWS = 12

EXPECTED_OBSERVED_SUCCESSES = 1479
EXPECTED_OBSERVED_TRIALS = 1500

def load_data():
    if not MARGINAL_FILENAME.is_file():
        raise FileNotFoundError(
            "Generation-success marginal "
            f"probabilities not found: "
            f"{MARGINAL_FILENAME}"
        )

    if not CONTRAST_FILENAME.is_file():
        raise FileNotFoundError(
            "Generation-success contrasts "
            f"not found: {CONTRAST_FILENAME}"
        )

    marginal = pd.read_csv(
        MARGINAL_FILENAME
    )

    contrasts = pd.read_csv(
        CONTRAST_FILENAME
    )

    return marginal, contrasts

def validate_marginal_probabilities(
    dataframe,
):
    required_columns = {
        "model",
        "representation",
        "posterior_mean",
        "posterior_median",
        "credible_interval_025",
        "credible_interval_975",
        "observed_successes",
        "observed_trials",
        "observed_success_rate",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Marginal probabilities missing "
            f"columns: {sorted(missing_columns)}"
        )

    if len(dataframe) != EXPECTED_MARGINAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_MARGINAL_ROWS} "
            f"marginal rows, found "
            f"{len(dataframe)}."
        )

    if dataframe[
        [
            "model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate marginal probability rows."
        )

    if set(
        dataframe["model"]
    ) != set(MODEL_LEVELS):
        raise ValueError(
            "Unexpected LLM levels."
        )

    if set(
        dataframe["representation"]
    ) != set(REPRESENTATION_LEVELS):
        raise ValueError(
            "Unexpected representation levels."
        )

    probability_columns = [
        "posterior_mean",
        "posterior_median",
        "credible_interval_025",
        "credible_interval_975",
        "observed_success_rate",
    ]

    for column in probability_columns:
        if not dataframe[
            column
        ].between(
            0.0,
            1.0,
            inclusive="both",
        ).all():
            raise ValueError(
                f"{column} contains values "
                "outside [0, 1]."
            )

    if not (
        dataframe[
            "credible_interval_025"
        ]
        <= dataframe[
            "posterior_median"
        ]
    ).all():
        raise ValueError(
            "Posterior median below lower "
            "credible limit."
        )

    if not (
        dataframe[
            "posterior_median"
        ]
        <= dataframe[
            "credible_interval_975"
        ]
    ).all():
        raise ValueError(
            "Posterior median above upper "
            "credible limit."
        )

    if (
        dataframe["observed_trials"] <= 0
    ).any():
        raise ValueError(
            "Non-positive observed trial count."
        )

    if (
        dataframe["observed_successes"] < 0
    ).any():
        raise ValueError(
            "Negative observed success count."
        )

    if (
        dataframe["observed_successes"]
        > dataframe["observed_trials"]
    ).any():
        raise ValueError(
            "Observed successes exceed trials."
        )

    reconstructed_rate = (
        dataframe["observed_successes"]
        / dataframe["observed_trials"]
    )

    if not np.allclose(
        dataframe[
            "observed_success_rate"
        ],
        reconstructed_rate,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "Observed success rate does not "
            "match successes / trials."
        )

    if int(
        dataframe[
            "observed_successes"
        ].sum()
    ) != EXPECTED_OBSERVED_SUCCESSES:
        raise ValueError(
            "Marginal table does not preserve "
            "1479 generation successes."
        )

    if int(
        dataframe[
            "observed_trials"
        ].sum()
    ) != EXPECTED_OBSERVED_TRIALS:
        raise ValueError(
            "Marginal table does not preserve "
            "1500 generation trials."
        )

    if not (
        dataframe[
            "observed_trials"
        ] == 100
    ).all():
        raise ValueError(
            "Every LLM × representation group "
            "must contain 100 observed trials."
        )

def validate_primary_contrasts(
    dataframe,
):
    required_columns = {
        "model",
        "representation",
        "reference",
        "posterior_mean_difference",
        "posterior_median_difference",
        "credible_interval_025",
        "credible_interval_975",
        "probability_difference_gt_0",
        "probability_difference_lt_0",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Primary contrasts missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != EXPECTED_CONTRAST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CONTRAST_ROWS} "
            f"contrast rows, found "
            f"{len(dataframe)}."
        )

    if dataframe[
        [
            "model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate generation-success "
            "contrast rows."
        )

    if not (
        dataframe["reference"]
        == REFERENCE_REPRESENTATION
    ).all():
        raise ValueError(
            "Generation-success contrast "
            "reference is not UML."
        )

    expected_representations = set(
        REPRESENTATION_LEVELS
    ) - {
        REFERENCE_REPRESENTATION
    }

    if set(
        dataframe["representation"]
    ) != expected_representations:
        raise ValueError(
            "Unexpected comparison "
            "representations."
        )


    probability_sum = (
        dataframe[
            "probability_difference_gt_0"
        ]
        + dataframe[
            "probability_difference_lt_0"
        ]
    )

    if not np.allclose(
        probability_sum,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Posterior direction probabilities "
            "do not sum to one."
        )

    if not (
        dataframe[
            "credible_interval_025"
        ]
        <= dataframe[
            "posterior_median_difference"
        ]
    ).all():
        raise ValueError(
            "Contrast median below lower "
            "credible limit."
        )

    if not (
        dataframe[
            "posterior_median_difference"
        ]
        <= dataframe[
            "credible_interval_975"
        ]
    ).all():
        raise ValueError(
            "Contrast median above upper "
            "credible limit."
        )

def build_long_results_table(
    marginal,
    contrasts,
):
    result = marginal.merge(
        contrasts[
            [
                "model",
                "representation",
                "reference",
                "posterior_mean_difference",
                "posterior_median_difference",
                "credible_interval_025",
                "credible_interval_975",
                "probability_difference_gt_0",
                "probability_difference_lt_0",
            ]
        ],
        on=[
            "model",
            "representation",
        ],
        how="left",
        validate="one_to_one",
        suffixes=(
            "_probability",
            "_difference",
        ),
    )

    if len(result) != EXPECTED_MARGINAL_ROWS:
        raise ValueError(
            "Long result merge changed "
            "the number of rows."
        )

    uml_rows = (
        result["representation"]
        == REFERENCE_REPRESENTATION
    )

    contrast_columns = [
        "reference",
        "posterior_mean_difference",
        "posterior_median_difference",
        "credible_interval_025_difference",
        "credible_interval_975_difference",
        "probability_difference_gt_0",
        "probability_difference_lt_0",
    ]

    if result.loc[
        uml_rows,
        contrast_columns,
    ].notna().any().any():
        raise ValueError(
            "UML reference rows unexpectedly "
            "contain contrast results."
        )

    comparison_rows = ~uml_rows

    if result.loc[
        comparison_rows,
        contrast_columns,
    ].isna().any().any():
        raise ValueError(
            "A non-UML representation is "
            "missing its contrast result."
        )

    return result

def format_probability(
    mean,
    lower,
    upper,
):
    return (
        f"{mean:.3f} "
        f"[{lower:.3f}, {upper:.3f}]"
    )


def format_signed_difference(
    mean,
    lower,
    upper,
):
    return (
        f"{mean:+.3f} "
        f"[{lower:+.3f}, {upper:+.3f}]"
    )

def format_result_cell(
    row,
):
    probability = format_probability(
        row["posterior_mean"],
        row[
            "credible_interval_025_probability"
        ],
        row[
            "credible_interval_975_probability"
        ],
    )

    observed = (
        f"{int(row['observed_successes'])}"
        f"/{int(row['observed_trials'])}"
    )

    if (
        row["representation"]
        == REFERENCE_REPRESENTATION
    ):
        return (
            f"{probability}; "
            f"observed {observed}"
        )

    difference = format_signed_difference(
        row[
            "posterior_mean_difference"
        ],
        row[
            "credible_interval_025_difference"
        ],
        row[
            "credible_interval_975_difference"
        ],
    )

    return (
        f"{probability}; "
        f"observed {observed}; "
        f"Δ={difference}"
    )

def format_result_cell(
    row,
):
    probability = format_probability(
        row["posterior_mean"],
        row[
            "credible_interval_025_probability"
        ],
        row[
            "credible_interval_975_probability"
        ],
    )

    observed = (
        f"{int(row['observed_successes'])}"
        f"/{int(row['observed_trials'])}"
    )

    if (
        row["representation"]
        == REFERENCE_REPRESENTATION
    ):
        return (
            f"{probability}; "
            f"observed {observed}"
        )

    difference = format_signed_difference(
        row[
            "posterior_mean_difference"
        ],
        row[
            "credible_interval_025_difference"
        ],
        row[
            "credible_interval_975_difference"
        ],
    )

    return (
        f"{probability}; "
        f"observed {observed}; "
        f"Δ={difference}"
    )

def build_results_matrix(
    long_results,
):
    rows = []

    for model_name in MODEL_LEVELS:
        model_data = long_results[
            long_results["model"]
            == model_name
        ]

        if len(model_data) != len(
            REPRESENTATION_LEVELS
        ):
            raise ValueError(
                f"Unexpected representation count "
                f"for {model_name}."
            )

        row = {
            "model": model_name,
        }

        for representation_name in (
            REPRESENTATION_LEVELS
        ):
            representation_rows = (
                model_data[
                    model_data[
                        "representation"
                    ]
                    == representation_name
                ]
            )

            if len(
                representation_rows
            ) != 1:
                raise ValueError(
                    "Expected exactly one "
                    f"{model_name} / "
                    f"{representation_name} row."
                )

            row[
                representation_name
            ] = format_result_cell(
                representation_rows.iloc[0]
            )

        rows.append(
            row
        )

    matrix = pd.DataFrame(
        rows
    )

    if len(matrix) != len(
        MODEL_LEVELS
    ):
        raise ValueError(
            "Unexpected result matrix size."
        )

    return matrix

def build_compact_contrast_table(
    contrasts,
):
    return contrasts[
        [
            "model",
            "representation",
            "reference",
            "posterior_mean_difference",
            "posterior_median_difference",
            "credible_interval_025",
            "credible_interval_975",
            "probability_difference_gt_0",
            "probability_difference_lt_0",
        ]
    ].copy()

def validate_output_tables(
    long_results,
    matrix,
    compact_contrasts,
):
    if len(long_results) != 15:
        raise ValueError(
            "Long generation-success table "
            "must contain 15 rows."
        )

    if len(matrix) != 3:
        raise ValueError(
            "Generation-success matrix "
            "must contain 3 rows."
        )

    if len(compact_contrasts) != 12:
        raise ValueError(
            "Generation-success contrast table "
            "must contain 12 rows."
        )

    if long_results[
        [
            "model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate long-table rows."
        )

    if matrix["model"].duplicated().any():
        raise ValueError(
            "Duplicate model rows in matrix."
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
    marginal, contrasts = load_data()

    validate_marginal_probabilities(
        marginal
    )

    validate_primary_contrasts(
        contrasts
    )

    long_results = (
        build_long_results_table(
            marginal,
            contrasts,
        )
    )

    matrix = (
        build_results_matrix(
            long_results
        )
    )

    compact_contrasts = (
        build_compact_contrast_table(
            contrasts
        )
    )

    validate_output_tables(
        long_results,
        matrix,
        compact_contrasts,
    )

    long_filename = write_table(
        long_results,
        "generation_success_results_long.csv",
    )

    matrix_filename = write_table(
        matrix,
        "generation_success_results_table.csv",
    )

    contrast_filename = write_table(
        compact_contrasts,
        "generation_success_contrasts.csv",
    )

    print(
        "Generation-success results tables "
        "prepared and validated successfully."
    )

    print(
        f"long_rows: {len(long_results)}"
    )

    print(
        f"table_rows: {len(matrix)}"
    )

    print(
        f"contrast_rows: "
        f"{len(compact_contrasts)}"
    )

    print()

    print(
        "Generation-success results:"
    )

    print(
        matrix.to_string(
            index=False
        )
    )

    print()

    print(
        "Primary contrasts:"
    )

    print(
        compact_contrasts.to_string(
            index=False
        )
    )

    print()

    print(
        f"saved: {long_filename}"
    )

    print(
        f"saved: {matrix_filename}"
    )

    print(
        f"saved: {contrast_filename}"
    )


if __name__ == "__main__":
    main()