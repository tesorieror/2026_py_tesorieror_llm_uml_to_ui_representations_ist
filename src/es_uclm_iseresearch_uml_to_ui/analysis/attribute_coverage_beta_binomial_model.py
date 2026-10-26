from pathlib import Path

import numpy as np
import pymc as pm

from es_uclm_iseresearch_uml_to_ui.analysis.attribute_coverage_model import (
    MODEL_LEVELS,
    REPRESENTATION_LEVELS,
    UML_LEVELS,
    RANDOM_SEED,
    load_data,
    validate_data,
    encode_data,
)


RESULTS_PATH = Path(
    "experiment/analysis/results/"
    "attribute_coverage_beta_binomial"
)

PRIOR_PREDICTIVE_DRAWS = 1000

def build_attribute_coverage_beta_binomial_model(
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

        alpha_beta = pm.Deterministic(
            "alpha_beta_binomial",
            probability * concentration,
            dims="observation",
        )

        beta_beta = pm.Deterministic(
            "beta_beta_binomial",
            (1.0 - probability)
            * concentration,
            dims="observation",
        )

        pm.BetaBinomial(
            "numerator_obs",
            n=denominator,
            alpha=alpha_beta,
            beta=beta_beta,
            observed=numerator,
            dims="observation",
        )

    return model

def sample_prior_predictive(
    model,
):
    with model:
        prior_predictive = (
            pm.sample_prior_predictive(
                samples=PRIOR_PREDICTIVE_DRAWS,
                random_seed=RANDOM_SEED,
            )
        )

    return prior_predictive


def summarize_prior_predictive(
    dataframe,
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
            "numerator_obs"
        ]
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

    if simulated.shape[1] != len(
        dataframe
    ):
        raise ValueError(
            "Prior-predictive observations "
            "do not match analytical data."
        )

    zero_coverage = (
        simulated == 0
    ).sum(axis=1)

    perfect_coverage = (
        simulated
        == denominator[None, :]
    ).sum(axis=1)

    attempt_coverage = (
        simulated
        / denominator[None, :]
    ).mean(axis=1)

    element_weighted_coverage = (
        simulated.sum(axis=1)
        / denominator.sum()
    )

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
            np.quantile(
                overdispersion,
                0.05,
            )
        ),
        "overdispersion_q50": float(
            np.quantile(
                overdispersion,
                0.50,
            )
        ),
        "overdispersion_q95": float(
            np.quantile(
                overdispersion,
                0.95,
            )
        ),

        "prior_zero_coverage_q025": float(
            np.quantile(
                zero_coverage,
                0.025,
            )
        ),
        "prior_zero_coverage_q50": float(
            np.quantile(
                zero_coverage,
                0.50,
            )
        ),
        "prior_zero_coverage_q975": float(
            np.quantile(
                zero_coverage,
                0.975,
            )
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
    prior_predictive,
):
    RESULTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        RESULTS_PATH
        / "prior_predictive.nc"
    )

    prior_predictive.to_netcdf(
        filename
    )

    return filename

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

    prior_predictive = (
        sample_prior_predictive(
            model
        )
    )

    prior_summary = (
        summarize_prior_predictive(
            dataframe,
            prior_predictive,
        )
    )

    filename = (
        save_prior_predictive(
            prior_predictive
        )
    )

    print(
        "Attribute-coverage beta-binomial "
        "prior predictive completed successfully."
    )

    print(
        f"Saved: {filename}"
    )

    for key, value in (
        prior_summary.items()
    ):
        print(
            f"{key}: {value:.6f}"
        )


if __name__ == "__main__":
    main()