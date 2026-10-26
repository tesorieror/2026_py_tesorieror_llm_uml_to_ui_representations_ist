from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

DATA_FILENAME = Path(
    "experiment/analysis/data/"
    "attribute_coverage.csv"
)

RESULTS_PATH = Path(
    "experiment/analysis/results/"
    "attribute_coverage"
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

PRIOR_PREDICTIVE_DRAWS = 1000

RANDOM_SEED = 1977


def load_data():
    if not DATA_FILENAME.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILENAME}"
        )

    return pd.read_csv(DATA_FILENAME)


def validate_data(dataframe):
    required_columns = {
        "model",
        "uml_model",
        "representation",
        "iteration",
        "cell_id",
        "numerator",
        "denominator",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != 1479:
        raise ValueError(
            "Expected 1479 evaluable attempts, "
            f"found {len(dataframe)}."
        )

    if dataframe[
        [
            "model",
            "uml_model",
            "representation",
            "iteration",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate experimental attempts."
        )

    if set(dataframe["model"]) != set(
        MODEL_LEVELS
    ):
        raise ValueError(
            "Unexpected model levels."
        )

    if set(
        dataframe["representation"]
    ) != set(REPRESENTATION_LEVELS):
        raise ValueError(
            "Unexpected representation levels."
        )

    if set(dataframe["uml_model"]) != set(
        UML_LEVELS
    ):
        raise ValueError(
            "Unexpected UML levels."
        )

    if (
        dataframe["denominator"] <= 0
    ).any():
        raise ValueError(
            "All denominators must be positive."
        )

    if (
        dataframe["numerator"] < 0
    ).any():
        raise ValueError(
            "Numerators cannot be negative."
        )

    if (
        dataframe["numerator"]
        > dataframe["denominator"]
    ).any():
        raise ValueError(
            "Numerator exceeds denominator."
        )

    denominators_per_uml = (
        dataframe
        .groupby("uml_model")[
            "denominator"
        ]
        .nunique()
    )

    if not (
        denominators_per_uml == 1
    ).all():
        raise ValueError(
            "Attribute denominator is not "
            "constant within UML model."
        )

def encode_data(dataframe):
    dataframe = dataframe.copy()

    model_to_index = {
        value: index
        for index, value
        in enumerate(MODEL_LEVELS)
    }

    representation_to_index = {
        value: index
        for index, value
        in enumerate(
            REPRESENTATION_LEVELS
        )
    }

    uml_to_index = {
        value: index
        for index, value
        in enumerate(UML_LEVELS)
    }

    cell_levels = sorted(
        dataframe["cell_id"].unique()
    )

    cell_to_index = {
        value: index
        for index, value
        in enumerate(cell_levels)
    }

    dataframe["model_index"] = (
        dataframe["model"].map(
            model_to_index
        )
    )

    dataframe[
        "representation_index"
    ] = dataframe["representation"].map(
        representation_to_index
    )

    dataframe["uml_index"] = (
        dataframe["uml_model"].map(
            uml_to_index
        )
    )

    dataframe["cell_index"] = (
        dataframe["cell_id"].map(
            cell_to_index
        )
    )

    if dataframe[
        [
            "model_index",
            "representation_index",
            "uml_index",
            "cell_index",
        ]
    ].isna().any().any():
        raise ValueError(
            "Failed to encode categorical "
            "variables."
        )

    return dataframe, cell_levels


def build_attribute_coverage_binomial_model(
    dataframe,
    cell_levels,
):
    model_index = dataframe[
        "model_index"
    ].to_numpy(dtype=int)

    representation_index = dataframe[
        "representation_index"
    ].to_numpy(dtype=int)

    uml_index = dataframe[
        "uml_index"
    ].to_numpy(dtype=int)

    cell_index = dataframe[
        "cell_index"
    ].to_numpy(dtype=int)

    numerator = dataframe[
        "numerator"
    ].to_numpy(dtype=int)

    denominator = dataframe[
        "denominator"
    ].to_numpy(dtype=int)

    coords = {
        "observation": np.arange(
            len(dataframe)
        ),
        "model_nonreference": (
            MODEL_LEVELS[1:]
        ),
        "representation_nonreference": (
            REPRESENTATION_LEVELS[1:]
        ),
        "interaction_model": (
            MODEL_LEVELS[1:]
        ),
        "interaction_representation": (
            REPRESENTATION_LEVELS[1:]
        ),
        "uml_model": UML_LEVELS,
        "cell": cell_levels,
    }

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal(
            "intercept",
            mu=0.0,
            sigma=1.5,
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
            sigma=0.5,
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
            sigma_uml * uml_raw,
            dims="uml_model",
        )

        sigma_cell = pm.HalfNormal(
            "sigma_cell",
            sigma=1.0,
        )

        cell_raw = pm.Normal(
            "cell_raw",
            mu=0.0,
            sigma=1.0,
            dims="cell",
        )

        cell_effect = pm.Deterministic(
            "cell_effect",
            sigma_cell * cell_raw,
            dims="cell",
        )

        eta = (
            intercept
            + uml_effect[uml_index]
            + cell_effect[cell_index]
        )

        eta = eta + pm.math.switch(
            pm.math.gt(
                model_index,
                0,
            ),
            model_effect_raw[
                pm.math.maximum(
                    model_index - 1,
                    0,
                )
            ],
            0.0,
        )

        eta = eta + pm.math.switch(
            pm.math.gt(
                representation_index,
                0,
            ),
            representation_effect_raw[
                pm.math.maximum(
                    representation_index - 1,
                    0,
                )
            ],
            0.0,
        )

        interaction_model_index = (
            pm.math.maximum(
                model_index - 1,
                0,
            )
        )

        interaction_representation_index = (
            pm.math.maximum(
                representation_index - 1,
                0,
            )
        )

        interaction_value = interaction_raw[
            interaction_model_index,
            interaction_representation_index,
        ]

        eta = eta + pm.math.switch(
            pm.math.and_(
                pm.math.gt(
                    model_index,
                    0,
                ),
                pm.math.gt(
                    representation_index,
                    0,
                ),
            ),
            interaction_value,
            0.0,
        )

        probability = pm.Deterministic(
            "probability",
            pm.math.sigmoid(eta),
            dims="observation",
        )

        pm.Binomial(
            "numerator_obs",
            n=denominator,
            p=probability,
            observed=numerator,
            dims="observation",
        )

    return model


def sample_prior_predictive(
    model,
):
    with model:
        inference_data = (
            pm.sample_prior_predictive(
                samples=PRIOR_PREDICTIVE_DRAWS,
                random_seed=RANDOM_SEED,
            )
        )

    return inference_data

def summarize_prior_predictive(
    inference_data,
    dataframe,
):
    probability = (
        inference_data
        .prior["probability"]
        .values
        .reshape(-1)
    )

    numerator = (
        inference_data
        .prior_predictive["numerator_obs"]
        .stack(sample=("chain", "draw"))
        .transpose(
            "sample",
            "observation",
        )
        .values
    )

    denominator = dataframe[
        "denominator"
    ].to_numpy(dtype=int)

    if numerator.shape[1] != len(
        denominator
    ):
        raise ValueError(
            "Prior-predictive observations "
            "do not match analytical data."
        )

    zero_coverage = (
        numerator == 0
    ).sum(axis=1)

    perfect_coverage = (
        numerator
        == denominator[None, :]
    ).sum(axis=1)

    attempt_coverage = (
        numerator
        / denominator[None, :]
    ).mean(axis=1)

    element_weighted_coverage = (
        numerator.sum(axis=1)
        / denominator.sum()
    )

    return {
        "probability_mean": float(
            np.mean(probability)
        ),
        "probability_q01": float(
            np.quantile(probability, 0.01)
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
        "probability_q99": float(
            np.quantile(probability, 0.99)
        ),
        "prior_zero_coverage_q025": float(
            np.quantile(zero_coverage, 0.025)
        ),
        "prior_zero_coverage_q50": float(
            np.quantile(zero_coverage, 0.50)
        ),
        "prior_zero_coverage_q975": float(
            np.quantile(zero_coverage, 0.975)
        ),
        "prior_perfect_coverage_q025": float(
            np.quantile(
                perfect_coverage,
                0.025,
            )
        ),
        "prior_perfect_coverage_q50": float(
            np.quantile(
                perfect_coverage,
                0.50,
            )
        ),
        "prior_perfect_coverage_q975": float(
            np.quantile(
                perfect_coverage,
                0.975,
            )
        ),
        "prior_attempt_coverage_q025": float(
            np.quantile(
                attempt_coverage,
                0.025,
            )
        ),
        "prior_attempt_coverage_q50": float(
            np.quantile(
                attempt_coverage,
                0.50,
            )
        ),
        "prior_attempt_coverage_q975": float(
            np.quantile(
                attempt_coverage,
                0.975,
            )
        ),
        "prior_element_weighted_coverage_q025": float(
            np.quantile(
                element_weighted_coverage,
                0.025,
            )
        ),
        "prior_element_weighted_coverage_q50": float(
            np.quantile(
                element_weighted_coverage,
                0.50,
            )
        ),
        "prior_element_weighted_coverage_q975": float(
            np.quantile(
                element_weighted_coverage,
                0.975,
            )
        ),
    }

def save_prior_predictive(
    inference_data,
):
    RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        RESULTS_PATH
        / "prior_predictive.nc"
    )

    inference_data.to_netcdf(
        filename
    )

    return filename


# Posterior

POSTERIOR_DRAWS = 2000
TUNE_DRAWS = 2000
CHAINS = 4
TARGET_ACCEPT = 0.99
RANDOM_SEED = 1977


def sample_posterior(
    model,
):
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

def save_posterior(
    inference_data,
):
    RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        RESULTS_PATH
        / "posterior.nc"
    )

    inference_data.to_netcdf(
        filename
    )

    return filename


def summarize_posterior_diagnostics(
    inference_data,
):
    variable_names = [
        "intercept",
        "model_effect_raw",
        "representation_effect_raw",
        "interaction_raw",
        "sigma_uml",
        "uml_raw",
        "sigma_cell",
        "cell_raw",
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



def main():
    dataframe = load_data()

    validate_data(dataframe)

    dataframe, cell_levels = (
        encode_data(dataframe)
    )

    print(
        "Attribute-coverage data loaded "
        "and validated successfully."
    )

    print(
        f"observations: {len(dataframe)}"
    )

    print(
        f"cells: {len(cell_levels)}"
    )

    print(
        "perfect_coverage_observations: "
        f"{int((dataframe['numerator'] == dataframe['denominator']).sum())}"
    )

    print(
        "zero_coverage_observations: "
        f"{int((dataframe['numerator'] == 0).sum())}"
    )

    print(
        "denominators_by_uml:"
    )

    print(
        dataframe
        .groupby("uml_model")[
            "denominator"
        ]
        .first()
        .reindex(UML_LEVELS)
        .to_string()
    )

    model = (
        build_attribute_coverage_binomial_model(
            dataframe,
            cell_levels,
        )
    )

    prior_predictive = (
        sample_prior_predictive(
            model
        )
    )

    prior_summary = (
        summarize_prior_predictive(
            prior_predictive,
            dataframe,
        )
    )

    prior_filename = (
        save_prior_predictive(
            prior_predictive
        )
    )

    print(
        "Attribute-coverage binomial "
        "prior predictive completed "
        "successfully."
    )

    print(
        f"Saved: {prior_filename}"
    )

    for key, value in (
        prior_summary.items()
    ):
        print(
            f"{key}: {value:.6f}"
        )


        posterior = sample_posterior(
        model
    )

    posterior_filename = (
        save_posterior(
            posterior
        )
    )

    posterior_diagnostics = (
        summarize_posterior_diagnostics(
            posterior
        )
    )

    print(
        "Attribute-coverage binomial "
        "posterior completed successfully."
    )

    print(
        f"Saved posterior: "
        f"{posterior_filename}"
    )

    for key, value in (
        posterior_diagnostics.items()
    ):
        print(
            f"{key}: {value}"
        )

if __name__ == "__main__":
    main()