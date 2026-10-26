from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


GENERATION_DATA_FILENAME = Path(
    "experiment/analysis/data/generation_success.csv"
)

POSTERIOR_FILENAME = Path(
    "experiment/analysis/results/"
    "generation_success_beta_binomial/"
    "posterior.nc"
)

ANALYSIS_RESULTS_PATH = Path(
    "experiment/analysis/results/"
    "generation_success_beta_binomial"
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

UML_LEVELS = [
    f"uml{i}"
    for i in range(1, 11)
]


def load_inputs():
    if not GENERATION_DATA_FILENAME.is_file():
        raise FileNotFoundError(
            "Generation-success analytical dataset "
            f"not found: {GENERATION_DATA_FILENAME}"
        )

    if not POSTERIOR_FILENAME.is_file():
        raise FileNotFoundError(
            "Generation-success posterior not found: "
            f"{POSTERIOR_FILENAME}"
        )

    dataframe = pd.read_csv(
        GENERATION_DATA_FILENAME
    )

    inference_data = az.from_netcdf(
        POSTERIOR_FILENAME
    )

    return dataframe, inference_data

def validate_inputs(
    dataframe,
    inference_data,
):
    if len(dataframe) != 150:
        raise ValueError(
            "Expected 150 generation-success cells, "
            f"found {len(dataframe)}."
        )

    expected_cells = (
        len(MODEL_LEVELS)
        * len(REPRESENTATION_LEVELS)
        * len(UML_LEVELS)
    )

    if len(dataframe) != expected_cells:
        raise ValueError(
            "Unexpected factorial design size."
        )

    if dataframe[
        [
            "model",
            "representation",
            "uml_model",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate generation-success cells."
        )

    required_posterior_variables = {
        "intercept",
        "model_effect_raw",
        "representation_effect_raw",
        "interaction_raw",
        "sigma_uml",
        "uml_raw",
        "uml_effect",
        "overdispersion",
    }

    missing_variables = (
        required_posterior_variables
        - set(inference_data.posterior.data_vars)
    )

    if missing_variables:
        raise KeyError(
            "Missing posterior variables: "
            f"{sorted(missing_variables)}"
        )

  
def reconstruct_probabilities(
    inference_data,
):
    posterior = inference_data.posterior

    intercept = (
        posterior["intercept"]
        .stack(sample=("chain", "draw"))
        .transpose("sample")
        .values
    )

    model_effect = (
        posterior["model_effect_raw"]
        .stack(sample=("chain", "draw"))
        .transpose(
            "sample",
            "model_nonreference",
        )
        .values
    )

    representation_effect = (
        posterior["representation_effect_raw"]
        .stack(sample=("chain", "draw"))
        .transpose(
            "sample",
            "representation_nonreference",
        )
        .values
    )

    interaction = (
        posterior["interaction_raw"]
        .stack(sample=("chain", "draw"))
        .transpose(
            "sample",
            "interaction_model",
            "interaction_representation",
        )
        .values
    )

    uml_effect = (
        posterior["uml_effect"]
        .stack(sample=("chain", "draw"))
        .transpose(
            "sample",
            "uml_model",
        )
        .values
    )

    n_samples = intercept.shape[0]

    probabilities = np.empty(
        (
            n_samples,
            len(MODEL_LEVELS),
            len(REPRESENTATION_LEVELS),
            len(UML_LEVELS),
        ),
        dtype=float,
    )

    for model_index in range(
        len(MODEL_LEVELS)
    ):
        for representation_index in range(
            len(REPRESENTATION_LEVELS)
        ):
            eta = (
                intercept[:, None]
                + uml_effect
            )

            if model_index > 0:
                eta = (
                    eta
                    + model_effect[
                        :,
                        model_index - 1,
                    ][:, None]
                )

            if representation_index > 0:
                eta = (
                    eta
                    + representation_effect[
                        :,
                        representation_index - 1,
                    ][:, None]
                )

            if (
                model_index > 0
                and representation_index > 0
            ):
                eta = (
                    eta
                    + interaction[
                        :,
                        model_index - 1,
                        representation_index - 1,
                    ][:, None]
                )

            probabilities[
                :,
                model_index,
                representation_index,
                :,
            ] = (
                1.0
                / (
                    1.0
                    + np.exp(-eta)
                )
            )

    return probabilities


def validate_reconstructed_probabilities(
    dataframe,
    inference_data,
    probabilities,
):
    expected_shape = (
        8000,
        len(MODEL_LEVELS),
        len(REPRESENTATION_LEVELS),
        len(UML_LEVELS),
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            "Unexpected reconstructed-probability "
            "shape. "
            f"Expected {expected_shape}, "
            f"found {probabilities.shape}."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Reconstructed probabilities contain "
            "non-finite values."
        )

    if (
        (probabilities <= 0)
        | (probabilities >= 1)
    ).any():
        raise ValueError(
            "Reconstructed probabilities must lie "
            "strictly inside (0, 1)."
        )

    stored_probability = (
        inference_data
        .posterior["probability"]
        .stack(sample=("chain", "draw"))
        .transpose(
            "sample",
            "observation",
        )
        .values
    )

    model_to_index = {
        level: index
        for index, level
        in enumerate(MODEL_LEVELS)
    }

    representation_to_index = {
        level: index
        for index, level
        in enumerate(
            REPRESENTATION_LEVELS
        )
    }

    uml_to_index = {
        level: index
        for index, level
        in enumerate(UML_LEVELS)
    }

    reconstructed_observations = np.empty(
        stored_probability.shape,
        dtype=float,
    )

    for observation_index, row in (
        dataframe.reset_index(
            drop=True
        ).iterrows()
    ):
        model_index = model_to_index[
            row["model"]
        ]

        representation_index = (
            representation_to_index[
                row["representation"]
            ]
        )

        uml_index = uml_to_index[
            row["uml_model"]
        ]

        reconstructed_observations[
            :,
            observation_index,
        ] = probabilities[
            :,
            model_index,
            representation_index,
            uml_index,
        ]

    if not np.allclose(
        stored_probability,
        reconstructed_observations,
        rtol=1e-12,
        atol=1e-12,
    ):
        max_difference = float(
            np.max(
                np.abs(
                    stored_probability
                    - reconstructed_observations
                )
            )
        )

        raise ValueError(
            "Reconstructed probabilities do not "
            "match the probabilities stored by "
            "PyMC. "
            f"Maximum absolute difference: "
            f"{max_difference}"
        )

    return float(
        np.max(
            np.abs(
                stored_probability
                - reconstructed_observations
            )
        )
    )

def compute_marginal_probabilities(
    probabilities,
):
    if probabilities.ndim != 4:
        raise ValueError(
            "Expected a four-dimensional probability array."
        )

    if probabilities.shape[1:] != (
        len(MODEL_LEVELS),
        len(REPRESENTATION_LEVELS),
        len(UML_LEVELS),
    ):
        raise ValueError(
            "Unexpected dimensions in posterior "
            "probability array."
        )

    marginal_probabilities = (
        probabilities.mean(axis=3)
    )

    expected_shape = (
        probabilities.shape[0],
        len(MODEL_LEVELS),
        len(REPRESENTATION_LEVELS),
    )

    if (
        marginal_probabilities.shape
        != expected_shape
    ):
        raise ValueError(
            "Unexpected marginal-probability shape. "
            f"Expected {expected_shape}, "
            f"found "
            f"{marginal_probabilities.shape}."
        )

    if not np.isfinite(
        marginal_probabilities
    ).all():
        raise ValueError(
            "Marginal probabilities contain "
            "non-finite values."
        )

    if (
        (marginal_probabilities <= 0)
        | (marginal_probabilities >= 1)
    ).any():
        raise ValueError(
            "Marginal probabilities must lie "
            "strictly inside (0, 1)."
        )

    return marginal_probabilities

def summarize_marginal_probabilities(
    marginal_probabilities,
):
    rows = []

    for model_index, model_name in enumerate(
        MODEL_LEVELS
    ):
        for (
            representation_index,
            representation_name,
        ) in enumerate(
            REPRESENTATION_LEVELS
        ):
            draws = marginal_probabilities[
                :,
                model_index,
                representation_index,
            ]

            rows.append(
                {
                    "model": model_name,
                    "representation": (
                        representation_name
                    ),
                    "posterior_mean": float(
                        np.mean(draws)
                    ),
                    "posterior_median": float(
                        np.median(draws)
                    ),
                    "credible_interval_025": float(
                        np.quantile(
                            draws,
                            0.025,
                        )
                    ),
                    "credible_interval_975": float(
                        np.quantile(
                            draws,
                            0.975,
                        )
                    ),
                }
            )

    summary = pd.DataFrame(rows)

    if len(summary) != 15:
        raise ValueError(
            "Expected 15 marginal probability "
            f"summaries, found {len(summary)}."
        )

    if summary[
        [
            "model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate LLM × representation "
            "summaries found."
        )

    probability_columns = [
        "posterior_mean",
        "posterior_median",
        "credible_interval_025",
        "credible_interval_975",
    ]

    for column in probability_columns:
        if not (
            summary[column].between(
                0,
                1,
                inclusive="both",
            )
        ).all():
            raise ValueError(
                f"{column} contains values "
                "outside [0, 1]."
            )

    if (
        summary["credible_interval_025"]
        > summary["posterior_median"]
    ).any():
        raise ValueError(
            "Posterior median below lower "
            "credible bound."
        )

    if (
        summary["posterior_median"]
        > summary["credible_interval_975"]
    ).any():
        raise ValueError(
            "Posterior median above upper "
            "credible bound."
        )

    return summary


def add_observed_success_rates(
    dataframe,
    summary,
):
    observed = (
        dataframe
        .groupby(
            [
                "model",
                "representation",
            ],
            as_index=False,
        )
        .agg(
            observed_successes=(
                "n_success",
                "sum",
            ),
            observed_trials=(
                "n_trials",
                "sum",
            ),
        )
    )

    observed["observed_success_rate"] = (
        observed["observed_successes"]
        / observed["observed_trials"]
    )

    if len(observed) != 15:
        raise ValueError(
            "Expected 15 observed "
            "LLM × representation groups."
        )

    result = summary.merge(
        observed,
        on=[
            "model",
            "representation",
        ],
        how="left",
        validate="one_to_one",
    )

    if result[
        "observed_success_rate"
    ].isna().any():
        raise ValueError(
            "Missing observed success rates "
            "after merge."
        )

    return result

def save_marginal_probability_summary(
    summary,
):
    ANALYSIS_RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        ANALYSIS_RESULTS_PATH
        / "marginal_probabilities.csv"
    )

    summary.to_csv(
        filename,
        index=False,
    )

    return filename


# Contrastes

PRIMARY_COMPARISON_REPRESENTATIONS = [
    "json",
    "xsd",
    "sql",
    "txt",
]


def compute_primary_contrasts(
    marginal_probabilities,
):
    uml_index = REPRESENTATION_LEVELS.index(
        "uml"
    )

    rows = []

    for model_index, model_name in enumerate(
        MODEL_LEVELS
    ):
        uml_draws = marginal_probabilities[
            :,
            model_index,
            uml_index,
        ]

        for representation_name in (
            PRIMARY_COMPARISON_REPRESENTATIONS
        ):
            representation_index = (
                REPRESENTATION_LEVELS.index(
                    representation_name
                )
            )

            representation_draws = (
                marginal_probabilities[
                    :,
                    model_index,
                    representation_index,
                ]
            )

            difference = (
                representation_draws
                - uml_draws
            )

            rows.append(
                {
                    "model": model_name,
                    "representation": (
                        representation_name
                    ),
                    "reference": "uml",
                    "posterior_mean_difference": float(
                        np.mean(difference)
                    ),
                    "posterior_median_difference": float(
                        np.median(difference)
                    ),
                    "credible_interval_025": float(
                        np.quantile(
                            difference,
                            0.025,
                        )
                    ),
                    "credible_interval_975": float(
                        np.quantile(
                            difference,
                            0.975,
                        )
                    ),
                    "probability_difference_gt_0": float(
                        np.mean(
                            difference > 0
                        )
                    ),
                    "probability_difference_lt_0": float(
                        np.mean(
                            difference < 0
                        )
                    ),
                }
            )

    contrasts = pd.DataFrame(rows)

    if len(contrasts) != 12:
        raise ValueError(
            "Expected 12 primary contrasts, "
            f"found {len(contrasts)}."
        )

    if contrasts[
        [
            "model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate primary contrasts found."
        )

    return contrasts

def validate_primary_contrasts(
    contrasts,
    marginal_summary,
):
    posterior_means = {
        (
            row["model"],
            row["representation"],
        ): row["posterior_mean"]
        for _, row in (
            marginal_summary.iterrows()
        )
    }

    maximum_difference = 0.0

    for _, row in contrasts.iterrows():
        expected = (
            posterior_means[
                (
                    row["model"],
                    row["representation"],
                )
            ]
            - posterior_means[
                (
                    row["model"],
                    "uml",
                )
            ]
        )

        actual = row[
            "posterior_mean_difference"
        ]

        difference = abs(
            actual - expected
        )

        maximum_difference = max(
            maximum_difference,
            difference,
        )

    if maximum_difference > 1e-12:
        raise ValueError(
            "Primary contrast means do not match "
            "differences between posterior means. "
            f"Maximum difference: "
            f"{maximum_difference}"
        )

    return maximum_difference

def save_primary_contrasts(
    contrasts,
):
    ANALYSIS_RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        ANALYSIS_RESULTS_PATH
        / "primary_contrasts.csv"
    )

    contrasts.to_csv(
        filename,
        index=False,
    )

    return filename



def main():
    dataframe, inference_data = load_inputs()

    validate_inputs(
        dataframe,
        inference_data,
    )

    print(
        "Generation-success posterior "
        "loaded and validated successfully."
    )

    print(
        f"cells: {len(dataframe)}"
    )

    print(
        "posterior_chains: "
        f"{inference_data.posterior.sizes['chain']}"
    )

    print(
        "posterior_draws_per_chain: "
        f"{inference_data.posterior.sizes['draw']}"
    )

    probabilities = (
        reconstruct_probabilities(
            inference_data
        )
    )

    max_difference = (
        validate_reconstructed_probabilities(
            dataframe,
            inference_data,
            probabilities,
        )
    )

    print(
        "Posterior probabilities reconstructed "
        "and validated successfully."
    )

    print(
        "probability_array_shape: "
        f"{probabilities.shape}"
    )

    print(
        "maximum_absolute_difference: "
        f"{max_difference:.16e}"
    )

    marginal_probabilities = (
        compute_marginal_probabilities(
            probabilities
        )
    )

    marginal_summary = (
        summarize_marginal_probabilities(
            marginal_probabilities
        )
    )

    marginal_summary = (
        add_observed_success_rates(
            dataframe,
            marginal_summary,
        )
    )

    marginal_filename = (
        save_marginal_probability_summary(
            marginal_summary
        )
    )

    print(
        "Marginal posterior probabilities "
        "computed successfully."
    )

    print(
        "marginal_probability_array_shape: "
        f"{marginal_probabilities.shape}"
    )

    print(
        f"Saved: {marginal_filename}"
    )

    print(
        marginal_summary.to_string(
            index=False
        )
    )

# Contrastes

    primary_contrasts = (
        compute_primary_contrasts(
            marginal_probabilities
        )
    )

    contrast_validation_difference = (
        validate_primary_contrasts(
            primary_contrasts,
            marginal_summary,
        )
    )

    contrasts_filename = (
        save_primary_contrasts(
            primary_contrasts
        )
    )

    print(
        "Primary representation contrasts "
        "computed and validated successfully."
    )

    print(
        "contrast_validation_max_difference: "
        f"{contrast_validation_difference:.16e}"
    )

    print(
        f"Saved: {contrasts_filename}"
    )

    print(
        primary_contrasts.to_string(
            index=False
        )
    )


  

if __name__ == "__main__":
    main()