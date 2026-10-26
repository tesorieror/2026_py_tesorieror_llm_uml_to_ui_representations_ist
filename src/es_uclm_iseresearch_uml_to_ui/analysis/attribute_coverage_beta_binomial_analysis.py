from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm

from es_uclm_iseresearch_uml_to_ui.analysis.attribute_coverage_model import (
    RANDOM_SEED,
    encode_data,
    load_data,
    validate_data,
)

from es_uclm_iseresearch_uml_to_ui.analysis.attribute_coverage_beta_binomial_model import (
    build_attribute_coverage_beta_binomial_model,
)


POSTERIOR_FILENAME = Path(
    "experiment/analysis/results/"
    "attribute_coverage_beta_binomial/"
    "posterior.nc"
)

RESULTS_PATH = Path(
    "experiment/analysis/results/"
    "attribute_coverage_beta_binomial"
)


def load_posterior():
    if not POSTERIOR_FILENAME.is_file():
        raise FileNotFoundError(
            "Beta-binomial posterior not found: "
            f"{POSTERIOR_FILENAME}"
        )

    return az.from_netcdf(
        POSTERIOR_FILENAME
    )


def validate_posterior(
    posterior,
    cell_levels,
):
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
        "overdispersion",
        "concentration",
        "alpha_beta_binomial",
        "beta_beta_binomial",
    }

    missing_variables = (
        required_variables
        - set(posterior.posterior.data_vars)
    )

    if missing_variables:
        raise KeyError(
            "Missing posterior variables: "
            f"{sorted(missing_variables)}"
        )

    if posterior.posterior.sizes["chain"] != 4:
        raise ValueError(
            "Expected 4 posterior chains."
        )

    if posterior.posterior.sizes["draw"] != 2000:
        raise ValueError(
            "Expected 2000 posterior draws "
            "per chain."
        )

    if posterior.posterior.sizes["cell"] != len(
        cell_levels
    ):
        raise ValueError(
            "Posterior cell dimension does not "
            "match analytical data."
        )

    divergences = int(
        posterior
        .sample_stats["diverging"]
        .sum()
        .values
    )

    if divergences != 0:
        raise ValueError(
            "Posterior contains "
            f"{divergences} divergences."
        )


def sample_posterior_predictive(
    dataframe,
    cell_levels,
    posterior,
):
    model = (
        build_attribute_coverage_beta_binomial_model(
            dataframe,
            cell_levels,
        )
    )

    with model:
        posterior_predictive = (
            pm.sample_posterior_predictive(
                posterior,
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
        .posterior_predictive["numerator_obs"]
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
    dataframe = load_data()

    validate_data(
        dataframe
    )

    dataframe, cell_levels = (
        encode_data(
            dataframe
        )
    )

    posterior = load_posterior()

    validate_posterior(
        posterior,
        cell_levels,
    )

    print(
        "Attribute-coverage beta-binomial "
        "posterior loaded and validated "
        "successfully."
    )

    print(
        f"observations: {len(dataframe)}"
    )

    print(
        f"cells: {len(cell_levels)}"
    )

    print(
        "posterior_chains: "
        f"{posterior.posterior.sizes['chain']}"
    )

    print(
        "posterior_draws_per_chain: "
        f"{posterior.posterior.sizes['draw']}"
    )

    posterior_predictive = (
        sample_posterior_predictive(
            dataframe,
            cell_levels,
            posterior,
        )
    )

    summary = (
        summarize_posterior_predictive(
            dataframe,
            posterior_predictive,
        )
    )

    filename = (
        save_posterior_predictive(
            posterior_predictive
        )
    )

    print(
        "Attribute-coverage beta-binomial "
        "posterior predictive completed "
        "successfully."
    )

    print(
        "Saved posterior predictive: "
        f"{filename}"
    )

    for key, value in (
        summary.items()
    ):
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()