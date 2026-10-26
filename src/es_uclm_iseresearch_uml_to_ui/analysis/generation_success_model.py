from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

GENERATION_DATA_FILENAME = Path(
    "experiment/analysis/data/generation_success.csv"
)

ANALYSIS_RESULTS_PATH = Path(
    "experiment/analysis/results/generation_success"
)

BETA_BINOMIAL_RESULTS_PATH = Path(
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


RANDOM_SEED = 1977
PRIOR_DRAWS = 4000

def load_generation_data():
    if not GENERATION_DATA_FILENAME.is_file():
        raise FileNotFoundError(
            "Generation-success analytical dataset "
            f"not found: {GENERATION_DATA_FILENAME}"
        )

    return pd.read_csv(
        GENERATION_DATA_FILENAME
    )


def validate_generation_data(dataframe):
    required_columns = {
        "model",
        "uml_model",
        "representation",
        "n_success",
        "n_trials",
        "n_not_success",
        "success_rate",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing generation-success columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != 150:
        raise ValueError(
            "Expected 150 generation-success cells, "
            f"found {len(dataframe)}."
        )

    if dataframe[
        [
            "model",
            "uml_model",
            "representation",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate generation-success cells found."
        )

    if set(dataframe["model"]) != set(MODEL_LEVELS):
        raise ValueError(
            "Unexpected LLM levels."
        )

    if (
        set(dataframe["representation"])
        != set(REPRESENTATION_LEVELS)
    ):
        raise ValueError(
            "Unexpected representation levels."
        )

    if (
        set(dataframe["uml_model"])
        != set(UML_LEVELS)
    ):
        raise ValueError(
            "Unexpected UML-model levels."
        )

    if not (dataframe["n_trials"] == 10).all():
        raise ValueError(
            "Every generation-success cell must "
            "contain exactly 10 trials."
        )

    if (dataframe["n_success"] < 0).any():
        raise ValueError(
            "Negative n_success found."
        )

    if (
        dataframe["n_success"]
        > dataframe["n_trials"]
    ).any():
        raise ValueError(
            "n_success exceeds n_trials."
        )

    if not np.array_equal(
        (
            dataframe["n_trials"]
            - dataframe["n_success"]
        ).to_numpy(),
        dataframe["n_not_success"].to_numpy(),
    ):
        raise ValueError(
            "n_not_success does not equal "
            "n_trials - n_success."
        )

    reconstructed_rate = (
        dataframe["n_success"]
        / dataframe["n_trials"]
    )

    if not np.allclose(
        dataframe["success_rate"],
        reconstructed_rate,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "success_rate does not equal "
            "n_success / n_trials."
        )

    if dataframe["n_success"].sum() != 1479:
        raise ValueError(
            "Expected 1479 successful generations."
        )

    if dataframe["n_not_success"].sum() != 21:
        raise ValueError(
            "Expected 21 failed generations."
        )

def encode_generation_data(dataframe):
    result = dataframe.copy()

    model_to_index = {
        level: index
        for index, level in enumerate(MODEL_LEVELS)
    }

    representation_to_index = {
        level: index
        for index, level
        in enumerate(REPRESENTATION_LEVELS)
    }

    uml_to_index = {
        level: index
        for index, level in enumerate(UML_LEVELS)
    }

    result["model_index"] = (
        result["model"]
        .map(model_to_index)
        .astype(int)
    )

    result["representation_index"] = (
        result["representation"]
        .map(representation_to_index)
        .astype(int)
    )

    result["uml_index"] = (
        result["uml_model"]
        .map(uml_to_index)
        .astype(int)
    )

    return result

def build_generation_model(dataframe):
    coords = {
        "observation": np.arange(len(dataframe)),
        "model_nonreference": MODEL_LEVELS[1:],
        "interaction_model": MODEL_LEVELS[1:],
        "interaction_representation": REPRESENTATION_LEVELS[1:],
        "representation_nonreference": (
            REPRESENTATION_LEVELS[1:]
        ),
        "uml_model": UML_LEVELS,
    }

    model_index = dataframe[
        "model_index"
    ].to_numpy()

    representation_index = dataframe[
        "representation_index"
    ].to_numpy()

    uml_index = dataframe[
        "uml_index"
    ].to_numpy()

    n_trials = dataframe[
        "n_trials"
    ].to_numpy(dtype=int)

    n_success = dataframe[
        "n_success"
    ].to_numpy(dtype=int)

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal(
            "intercept",
            mu=0.0,
            sigma=2.0,
        )

        model_effect_raw = pm.Normal(
            "model_effect_raw",
            mu=0.0,
            sigma=1.0,
            dims="model_nonreference",
        )

        representation_effect_raw = pm.Normal(
            "representation_effect_raw",
            mu=0.0,
            sigma=1.0,
            dims="representation_nonreference",
        )

        interaction_raw = pm.Normal(
            "interaction_raw",
            mu=0.0,
            sigma=0.75,
            dims=(
                "interaction_model",
                "interaction_representation",
            ),
        )

        sigma_uml = pm.HalfNormal(
            "sigma_uml",
            sigma=1.0,
        )

        uml_raw = pm.Normal(
            "uml_raw",
            mu=0.0,
            sigma=1.0,
            dims="uml_model",
        )

        uml_effect = pm.Deterministic(
            "uml_effect",
            uml_raw * sigma_uml,
            dims="uml_model",
        )

        eta = intercept + uml_effect[uml_index]

        model_nonreference = (
            model_index > 0
        )

        representation_nonreference = (
            representation_index > 0
        )

        eta = (
            eta
            + pm.math.switch(
                model_nonreference,
                model_effect_raw[
                    pm.math.maximum(
                        model_index - 1,
                        0,
                    )
                ],
                0.0,
            )
        )

        eta = (
            eta
            + pm.math.switch(
                representation_nonreference,
                representation_effect_raw[
                    pm.math.maximum(
                        representation_index - 1,
                        0,
                    )
                ],
                0.0,
            )
        )

        both_nonreference = (
            model_nonreference
            & representation_nonreference
        )

        interaction_value = interaction_raw[
            pm.math.maximum(
                model_index - 1,
                0,
            ),
            pm.math.maximum(
                representation_index - 1,
                0,
            ),
        ]

        eta = (
            eta
            + pm.math.switch(
                both_nonreference,
                interaction_value,
                0.0,
            )
        )

        probability = pm.Deterministic(
            "probability",
            pm.math.sigmoid(eta),
            dims="observation",
        )

        pm.Binomial(
            "n_success_obs",
            n=n_trials,
            p=probability,
            observed=n_success,
            dims="observation",
        )

    return model

def sample_prior_predictive(model):
    with model:
        prior_predictive = (
            pm.sample_prior_predictive(
                samples=PRIOR_DRAWS,
                random_seed=RANDOM_SEED,
            )
        )

    return prior_predictive


def summarize_prior_predictive(prior_predictive):
    probability = (
        prior_predictive
        .prior["probability"]
        .values
        .reshape(-1)
    )

    summary = {
        "probability_mean": float(
            np.mean(probability)
        ),
        "probability_sd": float(
            np.std(probability)
        ),
        "probability_q01": float(
            np.quantile(
                probability,
                0.01,
            )
        ),
        "probability_q05": float(
            np.quantile(
                probability,
                0.05,
            )
        ),
        "probability_q50": float(
            np.quantile(
                probability,
                0.50,
            )
        ),
        "probability_q95": float(
            np.quantile(
                probability,
                0.95,
            )
        ),
        "probability_q99": float(
            np.quantile(
                probability,
                0.99,
            )
        ),
    }

    return summary

def save_prior_predictive(
    prior_predictive,
):
    ANALYSIS_RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        ANALYSIS_RESULTS_PATH
        / "prior_predictive.nc"
    )

    prior_predictive.to_netcdf(
        filename
    )

    return filename


POSTERIOR_DRAWS = 2000
TUNE_DRAWS = 2000
CHAINS = 4
# TARGET_ACCEPT = 0.95
TARGET_ACCEPT = 0.99

def sample_posterior(model):
    with model:
        inference_data = pm.sample(
            draws=POSTERIOR_DRAWS,
            tune=TUNE_DRAWS,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
        )

    return inference_data

def save_posterior(inference_data):
    ANALYSIS_RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        ANALYSIS_RESULTS_PATH
        / "posterior.nc"
    )

    inference_data.to_netcdf(filename)

    return filename

def summarize_diagnostics(inference_data):
    variable_names = [
        "intercept",
        "model_effect_raw",
        "representation_effect_raw",
        "interaction_raw",
        "sigma_uml",
    ]

    summary = az.summary(
        inference_data,
        var_names=variable_names,
        round_to=None,
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
        inference_data
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

def sample_posterior_predictive(
    model,
    inference_data,
):
    with model:
        posterior_predictive = (
            pm.sample_posterior_predictive(
                inference_data,
                var_names=[
                    "n_success_obs",
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
            "n_success_obs"
        ]
        .values
    )

    simulated = simulated.reshape(
        -1,
        simulated.shape[-1],
    )

    observed = dataframe[
        "n_success"
    ].to_numpy(dtype=int)

    simulated_total_success = (
        simulated.sum(axis=1)
    )

    simulated_n_zero_cells = (
        simulated == 0
    ).sum(axis=1)

    simulated_n_full_cells = (
        simulated == 10
    ).sum(axis=1)

    simulated_n_below_full = (
        simulated < 10
    ).sum(axis=1)

    simulated_n_at_most_five = (
        simulated <= 5
    ).sum(axis=1)

    summary = {
        "observed_total_success": int(
            observed.sum()
        ),
        "observed_n_zero_cells": int(
            (observed == 0).sum()
        ),
        "observed_n_full_cells": int(
            (observed == 10).sum()
        ),
        "observed_n_below_full": int(
            (observed < 10).sum()
        ),
        "observed_n_at_most_five": int(
            (observed <= 5).sum()
        ),
        "pp_total_success_q025": float(
            np.quantile(
                simulated_total_success,
                0.025,
            )
        ),
        "pp_total_success_q50": float(
            np.quantile(
                simulated_total_success,
                0.50,
            )
        ),
        "pp_total_success_q975": float(
            np.quantile(
                simulated_total_success,
                0.975,
            )
        ),
        "pp_n_zero_cells_q025": float(
            np.quantile(
                simulated_n_zero_cells,
                0.025,
            )
        ),
        "pp_n_zero_cells_q50": float(
            np.quantile(
                simulated_n_zero_cells,
                0.50,
            )
        ),
        "pp_n_zero_cells_q975": float(
            np.quantile(
                simulated_n_zero_cells,
                0.975,
            )
        ),
        "pp_n_full_cells_q025": float(
            np.quantile(
                simulated_n_full_cells,
                0.025,
            )
        ),
        "pp_n_full_cells_q50": float(
            np.quantile(
                simulated_n_full_cells,
                0.50,
            )
        ),
        "pp_n_full_cells_q975": float(
            np.quantile(
                simulated_n_full_cells,
                0.975,
            )
        ),
        "pp_n_below_full_q025": float(
            np.quantile(
                simulated_n_below_full,
                0.025,
            )
        ),
        "pp_n_below_full_q50": float(
            np.quantile(
                simulated_n_below_full,
                0.50,
            )
        ),
        "pp_n_below_full_q975": float(
            np.quantile(
                simulated_n_below_full,
                0.975,
            )
        ),
        "pp_n_at_most_five_q025": float(
            np.quantile(
                simulated_n_at_most_five,
                0.025,
            )
        ),
        "pp_n_at_most_five_q50": float(
            np.quantile(
                simulated_n_at_most_five,
                0.50,
            )
        ),
        "pp_n_at_most_five_q975": float(
            np.quantile(
                simulated_n_at_most_five,
                0.975,
            )
        ),
    }

    return summary

def save_posterior_predictive(
    posterior_predictive,
):
    ANALYSIS_RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        ANALYSIS_RESULTS_PATH
        / "posterior_predictive.nc"
    )

    posterior_predictive.to_netcdf(
        filename
    )

    return filename




def build_generation_beta_binomial_model(dataframe):
    coords = {
        "observation": np.arange(len(dataframe)),
        "model_nonreference": MODEL_LEVELS[1:],
        "interaction_model": MODEL_LEVELS[1:],
        "interaction_representation": REPRESENTATION_LEVELS[1:],
        "representation_nonreference": REPRESENTATION_LEVELS[1:],
        "uml_model": UML_LEVELS,
    }

    model_index = dataframe["model_index"].to_numpy()
    representation_index = dataframe[
        "representation_index"
    ].to_numpy()
    uml_index = dataframe["uml_index"].to_numpy()

    n_trials = dataframe["n_trials"].to_numpy(dtype=int)
    n_success = dataframe["n_success"].to_numpy(dtype=int)

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal(
            "intercept",
            mu=0.0,
            sigma=2.0,
        )

        model_effect_raw = pm.Normal(
            "model_effect_raw",
            mu=0.0,
            sigma=1.0,
            dims="model_nonreference",
        )

        representation_effect_raw = pm.Normal(
            "representation_effect_raw",
            mu=0.0,
            sigma=1.0,
            dims="representation_nonreference",
        )

        interaction_raw = pm.Normal(
            "interaction_raw",
            mu=0.0,
            sigma=0.75,
            dims=(
                "interaction_model",
                "interaction_representation",
            ),
        )

        sigma_uml = pm.HalfNormal(
            "sigma_uml",
            sigma=1.0,
        )

        uml_raw = pm.Normal(
            "uml_raw",
            mu=0.0,
            sigma=1.0,
            dims="uml_model",
        )

        uml_effect = pm.Deterministic(
            "uml_effect",
            uml_raw * sigma_uml,
            dims="uml_model",
        )

        eta = intercept + uml_effect[uml_index]

        model_nonreference = model_index > 0
        representation_nonreference = (
            representation_index > 0
        )

        eta = (
            eta
            + pm.math.switch(
                model_nonreference,
                model_effect_raw[
                    pm.math.maximum(
                        model_index - 1,
                        0,
                    )
                ],
                0.0,
            )
        )

        eta = (
            eta
            + pm.math.switch(
                representation_nonreference,
                representation_effect_raw[
                    pm.math.maximum(
                        representation_index - 1,
                        0,
                    )
                ],
                0.0,
            )
        )

        both_nonreference = (
            model_nonreference
            & representation_nonreference
        )

        interaction_value = interaction_raw[
            pm.math.maximum(
                model_index - 1,
                0,
            ),
            pm.math.maximum(
                representation_index - 1,
                0,
            ),
        ]

        eta = (
            eta
            + pm.math.switch(
                both_nonreference,
                interaction_value,
                0.0,
            )
        )

        probability = pm.Deterministic(
            "probability",
            pm.math.sigmoid(eta),
            dims="observation",
        )

        overdispersion = pm.Beta(
            "overdispersion",
            alpha=1.0,
            beta=9.0,
        )

        concentration = pm.Deterministic(
            "concentration",
            (1.0 - overdispersion)
            / overdispersion,
        )

        alpha = pm.Deterministic(
            "alpha_beta_binomial",
            probability * concentration,
            dims="observation",
        )

        beta = pm.Deterministic(
            "beta_beta_binomial",
            (1.0 - probability)
            * concentration,
            dims="observation",
        )

        pm.BetaBinomial(
            "n_success_obs",
            n=n_trials,
            alpha=alpha,
            beta=beta,
            observed=n_success,
            dims="observation",
        )

    return model

def save_beta_binomial_prior_predictive(
    prior_predictive,
):
    BETA_BINOMIAL_RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        BETA_BINOMIAL_RESULTS_PATH
        / "prior_predictive.nc"
    )

    prior_predictive.to_netcdf(filename)

    return filename

def summarize_beta_binomial_prior_predictive(
    prior_predictive,
):
    probability = (
        prior_predictive
        .prior["probability"]
        .values
        .reshape(-1)
    )

    overdispersion = (
        prior_predictive
        .prior["overdispersion"]
        .values
        .reshape(-1)
    )

    simulated = (
        prior_predictive
        .prior_predictive[
            "n_success_obs"
        ]
        .values
    )

    simulated = simulated.reshape(
        -1,
        simulated.shape[-1],
    )

    simulated_n_zero_cells = (
        simulated == 0
    ).sum(axis=1)

    simulated_n_full_cells = (
        simulated == 10
    ).sum(axis=1)

    simulated_n_below_full = (
        simulated < 10
    ).sum(axis=1)

    return {
        "probability_mean": float(
            np.mean(probability)
        ),
        "probability_q05": float(
            np.quantile(probability, 0.05)
        ),
        "probability_q50": float(
            np.quantile(probability, 0.50)
        ),
        "probability_q95": float(
            np.quantile(probability, 0.95)
        ),

        "overdispersion_mean": float(
            np.mean(overdispersion)
        ),
        "overdispersion_q05": float(
            np.quantile(overdispersion, 0.05)
        ),
        "overdispersion_q50": float(
            np.quantile(overdispersion, 0.50)
        ),
        "overdispersion_q95": float(
            np.quantile(overdispersion, 0.95)
        ),

        "prior_n_zero_cells_q025": float(
            np.quantile(
                simulated_n_zero_cells,
                0.025,
            )
        ),
        "prior_n_zero_cells_q50": float(
            np.quantile(
                simulated_n_zero_cells,
                0.50,
            )
        ),
        "prior_n_zero_cells_q975": float(
            np.quantile(
                simulated_n_zero_cells,
                0.975,
            )
        ),

        "prior_n_full_cells_q025": float(
            np.quantile(
                simulated_n_full_cells,
                0.025,
            )
        ),
        "prior_n_full_cells_q50": float(
            np.quantile(
                simulated_n_full_cells,
                0.50,
            )
        ),
        "prior_n_full_cells_q975": float(
            np.quantile(
                simulated_n_full_cells,
                0.975,
            )
        ),

        "prior_n_below_full_q025": float(
            np.quantile(
                simulated_n_below_full,
                0.025,
            )
        ),
        "prior_n_below_full_q50": float(
            np.quantile(
                simulated_n_below_full,
                0.50,
            )
        ),
        "prior_n_below_full_q975": float(
            np.quantile(
                simulated_n_below_full,
                0.975,
            )
        ),
    }


def save_beta_binomial_posterior(
    inference_data,
):
    BETA_BINOMIAL_RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        BETA_BINOMIAL_RESULTS_PATH
        / "posterior.nc"
    )

    inference_data.to_netcdf(
        filename
    )

    return filename

def save_beta_binomial_posterior_predictive(
    posterior_predictive,
):
    BETA_BINOMIAL_RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        BETA_BINOMIAL_RESULTS_PATH
        / "posterior_predictive.nc"
    )

    posterior_predictive.to_netcdf(
        filename
    )

    return filename

def summarize_beta_binomial_diagnostics(
    inference_data,
):
    variable_names = [
        "intercept",
        "model_effect_raw",
        "representation_effect_raw",
        "interaction_raw",
        "sigma_uml",
        "overdispersion",
    ]

    summary = az.summary(
        inference_data,
        var_names=variable_names,
        round_to=None,
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
        inference_data
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
    inference_data,
):
    overdispersion = (
        inference_data
        .posterior["overdispersion"]
        .values
        .reshape(-1)
    )

    return {
        "overdispersion_mean": float(
            np.mean(overdispersion)
        ),
        "overdispersion_q025": float(
            np.quantile(
                overdispersion,
                0.025,
            )
        ),
        "overdispersion_q50": float(
            np.quantile(
                overdispersion,
                0.50,
            )
        ),
        "overdispersion_q975": float(
            np.quantile(
                overdispersion,
                0.975,
            )
        ),
    }


def main():
    dataframe = load_generation_data()

    validate_generation_data(
        dataframe
    )

    dataframe = encode_generation_data(
        dataframe
    )

    # model = build_generation_model(
    #     dataframe
    # )

    # prior_predictive = (
    #     sample_prior_predictive(
    #         model
    #     )
    # )

    # summary = (
    #     summarize_prior_predictive(
    #         prior_predictive
    #     )
    # )

    # filename = save_prior_predictive(
    #     prior_predictive
    # )

    # print(
    #     "Generation-success prior predictive "
    #     "completed successfully."
    # )

    # print(
    #     f"Saved: {filename}"
    # )

    # for key, value in summary.items():
    #     print(
    #         f"{key}: {value:.6f}"
    #     )

    # inference_data = sample_posterior(
    #     model
    # )

    # posterior_filename = save_posterior(
    #     inference_data
    # )

    # diagnostics = summarize_diagnostics(
    #     inference_data
    # )

    # print(
    #     f"Saved posterior: "
    #     f"{posterior_filename}"
    # )

    # for key, value in diagnostics.items():
    #     print(
    #         f"{key}: {value}"
    #     )

    # posterior_predictive = (
    #     sample_posterior_predictive(
    #         model,
    #         inference_data,
    #     )
    # )

    # posterior_predictive_filename = (
    #     save_posterior_predictive(
    #         posterior_predictive
    #     )
    # )

    # posterior_predictive_summary = (
    #     summarize_posterior_predictive(
    #         dataframe,
    #         posterior_predictive,
    #     )
    # )

    # print(
    #     "Saved posterior predictive: "
    #     f"{posterior_predictive_filename}"
    # )

    # for (
    #     key,
    #     value,
    # ) in posterior_predictive_summary.items():
    #     print(
    #         f"{key}: {value}"
    #     )

    beta_binomial_model = (
        build_generation_beta_binomial_model(
            dataframe
        )
    )

    beta_binomial_prior = (
        sample_prior_predictive(
            beta_binomial_model
        )
    )

    beta_binomial_summary = (
        summarize_beta_binomial_prior_predictive(
            beta_binomial_prior
        )
    )

    beta_filename = (
        save_beta_binomial_prior_predictive(
            beta_binomial_prior
        )
    )

    print(
        "Beta-binomial prior predictive completed."
    )

    print(
        f"Saved: {beta_filename}"
    )

    for key, value in beta_binomial_summary.items():
        print(
            f"{key}: {value:.6f}"
        )

    beta_binomial_inference_data = (
        sample_posterior(
            beta_binomial_model
        )
    )

    beta_binomial_posterior_filename = (
        save_beta_binomial_posterior(
            beta_binomial_inference_data
        )
    )

    beta_binomial_diagnostics = (
        summarize_beta_binomial_diagnostics(
            beta_binomial_inference_data
        )
    )

    overdispersion_summary = (
        summarize_overdispersion(
            beta_binomial_inference_data
        )
    )

    print(
        "Saved beta-binomial posterior: "
        f"{beta_binomial_posterior_filename}"
    )

    for key, value in beta_binomial_diagnostics.items():
        print(
            f"{key}: {value}"
        )

    for key, value in overdispersion_summary.items():
        print(
            f"{key}: {value}"
        )

    beta_binomial_posterior_predictive = (
        sample_posterior_predictive(
            beta_binomial_model,
            beta_binomial_inference_data,
        )
    )

    beta_binomial_pp_filename = (
        save_beta_binomial_posterior_predictive(
            beta_binomial_posterior_predictive
        )
    )

    beta_binomial_pp_summary = (
        summarize_posterior_predictive(
            dataframe,
            beta_binomial_posterior_predictive,
        )
    )

    print(
        "Saved beta-binomial posterior predictive: "
        f"{beta_binomial_pp_filename}"
    )

    for key, value in beta_binomial_pp_summary.items():
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()