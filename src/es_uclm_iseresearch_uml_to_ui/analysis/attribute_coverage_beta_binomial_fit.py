from pathlib import Path

import arviz as az
import pymc as pm
import numpy as np

from es_uclm_iseresearch_uml_to_ui.analysis.attribute_coverage_model import (
    load_data,
    validate_data,
    encode_data,
)

from es_uclm_iseresearch_uml_to_ui.analysis.attribute_coverage_beta_binomial_model import (
    build_attribute_coverage_beta_binomial_model,
)


RESULTS_PATH = Path(
    "experiment/analysis/results/"
    "attribute_coverage_beta_binomial"
)

POSTERIOR_DRAWS = 2000
TUNE_DRAWS = 2000
CHAINS = 4
TARGET_ACCEPT = 0.99
RANDOM_SEED = 1977


def sample_posterior(
    model,
):
    with model:
        posterior = pm.sample(
            draws=POSTERIOR_DRAWS,
            tune=TUNE_DRAWS,
            chains=CHAINS,
            cores=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
        )

    return posterior

def save_posterior(
    posterior,
):
    RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        RESULTS_PATH
        / "posterior.nc"
    )

    posterior.to_netcdf(
        filename
    )

    return filename

def summarize_posterior_diagnostics(
    posterior,
):
    summary = az.summary(
        posterior,
        var_names=[
            "intercept",
            "model_effect_raw",
            "representation_effect_raw",
            "interaction_raw",
            "sigma_uml",
            "sigma_cell",
            "overdispersion",
        ],
    )

    max_rhat = float(
        summary["r_hat"].max()
    )

    min_ess_bulk = float(
        summary["ess_bulk"].min()
    )

    min_ess_tail = float(
        summary["ess_tail"].min()
    )

    divergences = int(
        posterior
        .sample_stats["diverging"]
        .sum()
        .values
    )

    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "divergences": divergences,
    }

def summarize_overdispersion(
    posterior,
):
    values = (
        posterior
        .posterior["overdispersion"]
        .values
        .reshape(-1)
    )

    return {
        "overdispersion_mean": float(
            np.mean(values)
        ),
        "overdispersion_q025": float(
            np.quantile(values, 0.025)
        ),
        "overdispersion_q50": float(
            np.quantile(values, 0.50)
        ),
        "overdispersion_q975": float(
            np.quantile(values, 0.975)
        ),
    }

def main():
    dataframe = load_data()

    validate_data(dataframe)

    dataframe, cell_levels = (
        encode_data(dataframe)
    )

    print(
        "Attribute-coverage beta-binomial "
        "data loaded and validated successfully."
    )

    print(
        f"observations: {len(dataframe)}"
    )

    print(
        f"cells: {len(cell_levels)}"
    )

    model = (
        build_attribute_coverage_beta_binomial_model(
            dataframe,
            cell_levels,
        )
    )

    posterior = sample_posterior(
        model
    )

    filename = save_posterior(
        posterior
    )

    diagnostics = (
        summarize_posterior_diagnostics(
            posterior
        )
    )

    overdispersion_summary = (
        summarize_overdispersion(
            posterior
        )
    )

    print(
        "Attribute-coverage beta-binomial "
        "posterior completed successfully."
    )

    print(
        f"Saved posterior: {filename}"
    )

    for key, value in (
        diagnostics.items()
    ):
        print(
            f"{key}: {value}"
        )

    for key, value in (
        overdispersion_summary.items()
    ):
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()