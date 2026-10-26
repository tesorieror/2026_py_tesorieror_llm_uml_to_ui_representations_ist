from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from es_uclm_iseresearch_uml_to_ui.analysis.attribute_coverage_model import (
    MODEL_LEVELS,
    REPRESENTATION_LEVELS,
    UML_LEVELS,
    RANDOM_SEED,
    build_attribute_coverage_binomial_model,
    encode_data,
    validate_data,
)


DATA_FILENAME = Path(
    "experiment/analysis/data/"
    "attribute_coverage.csv"
)

POSTERIOR_FILENAME = Path(
    "experiment/analysis/results/"
    "attribute_coverage/"
    "posterior_validated.nc"
)

RESULTS_PATH = Path(
    "experiment/analysis/results/"
    "attribute_coverage"
)


def load_inputs():
    if not DATA_FILENAME.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILENAME}"
        )

    if not POSTERIOR_FILENAME.is_file():
        raise FileNotFoundError(
            "Validated posterior not found: "
            f"{POSTERIOR_FILENAME}"
        )

    dataframe = pd.read_csv(
        DATA_FILENAME
    )

    inference_data = az.from_netcdf(
        POSTERIOR_FILENAME
    )

    return dataframe, inference_data

def validate_inputs(
    dataframe,
    inference_data,
):
    validate_data(dataframe)

    required_variables = {
        "intercept",
        "model_effect_raw",
        "representation_effect_raw",
        "interaction_raw",
        "sigma_uml",
        "uml_raw",
        "uml_effect",
        "sigma_cell",
        "cell_raw",
        "cell_effect",
        "probability",
    }

    missing_variables = (
        required_variables
        - set(inference_data.posterior.data_vars)
    )

    if missing_variables:
        raise KeyError(
            "Missing posterior variables: "
            f"{sorted(missing_variables)}"
        )

    if inference_data.posterior.sizes[
        "chain"
    ] != 4:
        raise ValueError(
            "Expected 4 posterior chains."
        )

    if inference_data.posterior.sizes[
        "draw"
    ] != 2000:
        raise ValueError(
            "Expected 2000 posterior draws "
            "per chain."
        )

    divergences = int(
        inference_data
        .sample_stats["diverging"]
        .sum()
        .values
    )

    if divergences != 0:
        raise ValueError(
            "Validated posterior contains "
            f"{divergences} divergences."
        )

def sample_posterior_predictive(
    dataframe,
    cell_levels,
    inference_data,
):
    model = (
        build_attribute_coverage_binomial_model(
            dataframe,
            cell_levels,
        )
    )

    with model:
        posterior_predictive = (
            pm.sample_posterior_predictive(
                inference_data,
                var_names=[
                    "numerator_obs",
                ],
                random_seed=RANDOM_SEED,
                return_inferencedata=True,
            )
        )

    return posterior_predictive


def summarize_posterior_predictive(
    dataframe,
    posterior_predictive,
):
    simulated = (
        posterior_predictive
        .posterior_predictive[
            "numerator_obs"
        ]
        .stack(sample=("chain", "draw"))
        .transpose(
            "sample",
            "observation",
        )
        .values
    )

    observed_numerator = dataframe[
        "numerator"
    ].to_numpy(dtype=int)

    denominator = dataframe[
        "denominator"
    ].to_numpy(dtype=int)

    if simulated.shape[1] != len(
        dataframe
    ):
        raise ValueError(
            "Posterior-predictive observations "
            "do not match analytical data."
        )

    observed_zero = int(
        (observed_numerator == 0).sum()
    )

    observed_perfect = int(
        (
            observed_numerator
            == denominator
        ).sum()
    )

    observed_attempt_coverage = float(
        np.mean(
            observed_numerator
            / denominator
        )
    )

    observed_element_weighted_coverage = float(
        observed_numerator.sum()
        / denominator.sum()
    )

    simulated_zero = (
        simulated == 0
    ).sum(axis=1)

    simulated_perfect = (
        simulated
        == denominator[None, :]
    ).sum(axis=1)

    simulated_attempt_coverage = (
        simulated
        / denominator[None, :]
    ).mean(axis=1)

    simulated_element_weighted_coverage = (
        simulated.sum(axis=1)
        / denominator.sum()
    )

    return {
        "observed_zero_coverage": (
            observed_zero
        ),
        "observed_perfect_coverage": (
            observed_perfect
        ),
        "observed_attempt_coverage": (
            observed_attempt_coverage
        ),
        "observed_element_weighted_coverage": (
            observed_element_weighted_coverage
        ),

        "pp_zero_coverage_q025": float(
            np.quantile(
                simulated_zero,
                0.025,
            )
        ),
        "pp_zero_coverage_q50": float(
            np.quantile(
                simulated_zero,
                0.50,
            )
        ),
        "pp_zero_coverage_q975": float(
            np.quantile(
                simulated_zero,
                0.975,
            )
        ),

        "pp_perfect_coverage_q025": float(
            np.quantile(
                simulated_perfect,
                0.025,
            )
        ),
        "pp_perfect_coverage_q50": float(
            np.quantile(
                simulated_perfect,
                0.50,
            )
        ),
        "pp_perfect_coverage_q975": float(
            np.quantile(
                simulated_perfect,
                0.975,
            )
        ),

        "pp_attempt_coverage_q025": float(
            np.quantile(
                simulated_attempt_coverage,
                0.025,
            )
        ),
        "pp_attempt_coverage_q50": float(
            np.quantile(
                simulated_attempt_coverage,
                0.50,
            )
        ),
        "pp_attempt_coverage_q975": float(
            np.quantile(
                simulated_attempt_coverage,
                0.975,
            )
        ),

        "pp_element_weighted_coverage_q025": float(
            np.quantile(
                simulated_element_weighted_coverage,
                0.025,
            )
        ),
        "pp_element_weighted_coverage_q50": float(
            np.quantile(
                simulated_element_weighted_coverage,
                0.50,
            )
        ),
        "pp_element_weighted_coverage_q975": float(
            np.quantile(
                simulated_element_weighted_coverage,
                0.975,
            )
        ),
    }

def save_posterior_predictive(
    posterior_predictive,
):
    RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        RESULTS_PATH
        / "posterior_predictive.nc"
    )

    posterior_predictive.to_netcdf(
        filename
    )

    return filename



def main():
    dataframe, inference_data = (
        load_inputs()
    )

    validate_inputs(
        dataframe,
        inference_data,
    )

    dataframe, cell_levels = (
        encode_data(dataframe)
    )

    print(
        "Attribute-coverage posterior loaded "
        "and validated successfully."
    )

    print(
        f"observations: {len(dataframe)}"
    )

    print(
        f"cells: {len(cell_levels)}"
    )

    print(
        "posterior_chains: "
        f"{inference_data.posterior.sizes['chain']}"
    )

    print(
        "posterior_draws_per_chain: "
        f"{inference_data.posterior.sizes['draw']}"
    )

    posterior_predictive = (
        sample_posterior_predictive(
            dataframe,
            cell_levels,
            inference_data,
        )
    )

    posterior_predictive_summary = (
        summarize_posterior_predictive(
            dataframe,
            posterior_predictive,
        )
    )

    posterior_predictive_filename = (
        save_posterior_predictive(
            posterior_predictive
        )
    )

    print(
        "Attribute-coverage posterior "
        "predictive completed successfully."
    )

    print(
        "Saved posterior predictive: "
        f"{posterior_predictive_filename}"
    )

    for key, value in (
        posterior_predictive_summary.items()
    ):
        print(
            f"{key}: {value}"
        )



if __name__ == "__main__":
    main()