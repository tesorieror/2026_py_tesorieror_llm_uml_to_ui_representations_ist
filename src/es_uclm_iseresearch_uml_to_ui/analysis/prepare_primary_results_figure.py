from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILENAME = Path(
    "experiment/analysis/tables/"
    "primary_contrast_summary_long.csv"
)

FIGURE_PATH = Path(
    "experiment/analysis/figures"
)


OUTCOME_LEVELS = [
    "attribute_coverage",
    "primitive_type_fidelity",
    "enumeration_literal_coverage",
    "enumeration_cardinality_fidelity",
    "attribute_nonempty_label_coverage",
    "association_end_coverage",
    "association_cardinality_fidelity",
]


OUTCOME_LABELS = {
    "attribute_coverage": (
        "Attribute coverage"
    ),
    "primitive_type_fidelity": (
        "Primitive-type fidelity"
    ),
    "enumeration_literal_coverage": (
        "Enumeration-literal coverage"
    ),
    "enumeration_cardinality_fidelity": (
        "Enumeration-cardinality fidelity"
    ),
    "attribute_nonempty_label_coverage": (
        "Attribute non-empty-label coverage"
    ),
    "association_end_coverage": (
        "Association-end coverage"
    ),
    "association_cardinality_fidelity": (
        "Association-cardinality fidelity"
    ),
}


MODEL_LEVELS = [
    "llama3.1:8b",
    "mistral:7b-instruct",
    "qwen2.5:7b-instruct",
]


MODEL_LABELS = {
    "llama3.1:8b": "Llama 3.1 8B",
    "mistral:7b-instruct": (
        "Mistral 7B Instruct"
    ),
    "qwen2.5:7b-instruct": (
        "Qwen2.5 7B Instruct"
    ),
}


COMPARISON_LEVELS = [
    "json",
    "xsd",
    "sql",
    "txt",
]


COMPARISON_LABELS = {
    "json": "JSON",
    "xsd": "XSD",
    "sql": "SQL",
    "txt": "CNL",
}


EXPECTED_ROWS = 84

def load_data():
    if not INPUT_FILENAME.is_file():
        raise FileNotFoundError(
            f"Input table not found: "
            f"{INPUT_FILENAME}"
        )

    return pd.read_csv(
        INPUT_FILENAME
    )

def validate_data(
    dataframe,
):
    required_columns = {
        "outcome",
        "model",
        "comparison",
        "n_paired_umls",
        "analysis_status",
        "element_weighted_difference",
        "element_bb_mean",
        "element_bb_q025",
        "element_bb_q975",
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

    if len(dataframe) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(dataframe)}."
        )

    if dataframe[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate contrast rows."
        )

    if set(
        dataframe["outcome"]
    ) != set(OUTCOME_LEVELS):
        raise ValueError(
            "Unexpected outcome levels."
        )

    if set(
        dataframe["model"]
    ) != set(MODEL_LEVELS):
        raise ValueError(
            "Unexpected model levels."
        )

    if set(
        dataframe["comparison"]
    ) != set(COMPARISON_LEVELS):
        raise ValueError(
            "Unexpected comparison levels."
        )
    bootstrap = (
        dataframe["analysis_status"]
        == "bayesian_bootstrap"
    )

    if dataframe.loc[
        bootstrap,
        [
            "element_bb_mean",
            "element_bb_q025",
            "element_bb_q975",
        ],
    ].isna().any().any():
        raise ValueError(
            "Bootstrap contrast missing "
            "posterior summary."
        )

    descriptive = (
        dataframe["analysis_status"]
        == "descriptive_only"
    )

    if dataframe.loc[
        descriptive,
        "element_weighted_difference",
    ].isna().any():
        raise ValueError(
            "Descriptive contrast missing "
            "observed difference."
        )

    not_estimable = (
        dataframe["analysis_status"]
        == "not_estimable"
    )

    if not (
        dataframe.loc[
            not_estimable,
            "n_paired_umls",
        ] == 0
    ).all():
        raise ValueError(
            "Non-estimable contrast has "
            "non-zero support."
        )


def build_plot_data(
    dataframe,
):
    rows = []

    current_y = 0.0

    outcome_centers = {}
    separator_positions = []

    comparison_offsets = {
        "json": 0.27,
        "xsd": 0.09,
        "sql": -0.09,
        "txt": -0.27,
    }

    for outcome_index, outcome_name in enumerate(
        OUTCOME_LEVELS
    ):
        block_positions = []

        for model_name in MODEL_LEVELS:
            model_center = current_y

            subset = dataframe[
                (
                    dataframe["outcome"]
                    == outcome_name
                )
                & (
                    dataframe["model"]
                    == model_name
                )
            ]

            for comparison_name in (
                COMPARISON_LEVELS
            ):
                comparison_row = subset[
                    subset["comparison"]
                    == comparison_name
                ]

                if len(comparison_row) != 1:
                    raise ValueError(
                        "Missing plot contrast for "
                        f"{outcome_name} / "
                        f"{model_name} / "
                        f"{comparison_name}."
                    )

                source = (
                    comparison_row.iloc[0]
                )

                rows.append(
                    {
                        **source.to_dict(),
                        "y": (
                            model_center
                            + comparison_offsets[
                                comparison_name
                            ]
                        ),
                    }
                )

            block_positions.append(
                model_center
            )

            current_y += 1.0

        outcome_centers[
            outcome_name
        ] = float(
            np.mean(block_positions)
        )

        if outcome_index < (
            len(OUTCOME_LEVELS) - 1
        ):
            separator_positions.append(
                current_y - 0.5
            )

        current_y += 0.65

    return (
        pd.DataFrame(rows),
        outcome_centers,
        separator_positions,
    )

MARKERS = {
    "json": "o",
    "xsd": "s",
    "sql": "^",
    "txt": "D",
}


POINT_COLOR = "0.20"
INTERVAL_COLOR = "0.45"
SEPARATOR_COLOR = "0.85"
ZERO_LINE_COLOR = "0.35"
TEXT_COLOR = "0.15"





def create_figure(
    plot_data,
    outcome_centers,
    separator_positions,
):
    figure, axis = plt.subplots(
        figsize=(10.5, 14.5)
    )

    axis.axvline(
        0.0,
        linewidth=1.0,
        linestyle="--",
        color=ZERO_LINE_COLOR,
        zorder=0,
    )

    for separator in separator_positions:
        axis.axhline(
            separator,
            linewidth=0.6,
            color=SEPARATOR_COLOR,
            zorder=0,
        )

    for _, row in (
        plot_data.iterrows()
    ):
        status = row[
            "analysis_status"
        ]

        comparison = row[
            "comparison"
        ]

        marker = MARKERS[
            comparison
        ]

        if status == "bayesian_bootstrap":
            mean = row[
                "element_bb_mean"
            ]

            lower = row[
                "element_bb_q025"
            ]

            upper = row[
                "element_bb_q975"
            ]

            asymmetric_error = np.array(
                [
                    [
                        mean - lower
                    ],
                    [
                        upper - mean
                    ],
                ]
            )

            axis.errorbar(
                mean,
                row["y"],
                xerr=asymmetric_error,
                fmt=marker,
                markersize=5,
                capsize=2,
                linewidth=1.0,
                color=POINT_COLOR,
                ecolor=INTERVAL_COLOR,
                markerfacecolor=POINT_COLOR,
                markeredgecolor=POINT_COLOR,
                zorder=3,
            )

        elif status == "descriptive_only":
            axis.plot(
                row["element_weighted_difference"],
                row["y"],
                marker=marker,
                markersize=6,
                markerfacecolor="none",
                markeredgecolor=POINT_COLOR,
                markeredgewidth=1.2,
                linestyle="None",
                zorder=3,
            )

        elif status == "not_estimable":
            axis.text(
                0.03,
                row["y"],
                "NE",
                ha="left",
                va="center",
                fontsize=7,
                color=TEXT_COLOR,
                fontstyle="italic",
                zorder=3,
            )

        else:
            raise ValueError(
                f"Unexpected status: {status}"
            )

    model_tick_positions = []
    model_tick_labels = []

    for outcome_name in OUTCOME_LEVELS:
        outcome_rows = plot_data[
            plot_data["outcome"]
            == outcome_name
        ]

        for model_name in MODEL_LEVELS:
            model_rows = outcome_rows[
                outcome_rows["model"]
                == model_name
            ]

            model_tick_positions.append(
                float(
                    model_rows["y"].mean()
                )
            )

            model_tick_labels.append(
                MODEL_LABELS[
                    model_name
                ]
            )

    axis.set_yticks(
        model_tick_positions
    )

    axis.set_yticklabels(
        model_tick_labels,
        fontsize=8,
    )

   
    axis.relim()
    axis.autoscale_view()

    x_left, x_right = axis.get_xlim()

    outcome_label_x = (
        x_left
        + 0.01 * (x_right - x_left)
    )



    for outcome_name, center in outcome_centers.items():
        axis.text(
            outcome_label_x,
            center - 0.47,
            OUTCOME_LABELS[outcome_name],
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=TEXT_COLOR,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 1.5,
            },
            zorder=4,
        )

    legend_handles = []

    for comparison_name in COMPARISON_LEVELS:
        handle, = axis.plot(
            [],
            [],
            marker=MARKERS[comparison_name],
            linestyle="None",
            markerfacecolor=POINT_COLOR,
            markeredgecolor=POINT_COLOR,
            color=POINT_COLOR,
            label=COMPARISON_LABELS[
                comparison_name
            ],
        )

        legend_handles.append(
            handle
        )

    descriptive_handle, = axis.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor=POINT_COLOR,
        markeredgewidth=1.2,
        color=POINT_COLOR,
        label="Descriptive only",
    )

    legend_handles.append(
        descriptive_handle
    )


    axis.legend(
        handles=legend_handles,
        title="Representation / status",
        loc="lower right",
        frameon=False,
        ncol=2,
    )

    axis.grid(
        axis="x",
        linewidth=0.4,
        alpha=0.25,
    )

    axis.set_xlabel(
        "Difference from UML/XMI "
        "in element-weighted probability (Δ)"
    )

    axis.set_ylabel("")

    axis.invert_yaxis()

    figure.tight_layout()

    return figure

def save_figure(
    figure,
):
    FIGURE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_filename = (
        FIGURE_PATH
        / "primary_element_weighted_contrasts.png"
    )

    pdf_filename = (
        FIGURE_PATH
        / "primary_element_weighted_contrasts.pdf"
    )

    figure.savefig(
        png_filename,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_filename,
        bbox_inches="tight",
    )

    return (
        png_filename,
        pdf_filename,
    )

def validate_plot_data(
    plot_data,
):
    if len(plot_data) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} plot rows, "
            f"found {len(plot_data)}."
        )

    if plot_data[
        [
            "outcome",
            "model",
            "comparison",
        ]
    ].duplicated().any():
        raise ValueError(
            "Duplicate plot rows."
        )

    if plot_data[
        "y"
    ].isna().any():
        raise ValueError(
            "Missing vertical plot position."
        )

    if plot_data[
        "y"
    ].duplicated().any():
        raise ValueError(
            "Duplicate vertical plot positions."
        )

def main():
    dataframe = load_data()

    validate_data(
        dataframe
    )

    (
        plot_data,
        outcome_centers,
        separator_positions,
    ) = build_plot_data(
        dataframe
    )

    validate_plot_data(
        plot_data
    )

    figure = create_figure(
        plot_data,
        outcome_centers,
        separator_positions,
    )

    (
        png_filename,
        pdf_filename,
    ) = save_figure(
        figure
    )

    plt.close(
        figure
    )

    print(
        "Primary results figure "
        "prepared and validated successfully."
    )

    print(
        f"plot_rows: {len(plot_data)}"
    )

    print(
        "bayesian_bootstrap_rows: "
        f"{int((plot_data['analysis_status'] == 'bayesian_bootstrap').sum())}"
    )

    print(
        "descriptive_only_rows: "
        f"{int((plot_data['analysis_status'] == 'descriptive_only').sum())}"
    )

    print(
        "not_estimable_rows: "
        f"{int((plot_data['analysis_status'] == 'not_estimable').sum())}"
    )

    print(
        f"saved: {png_filename}"
    )

    print(
        f"saved: {pdf_filename}"
    )


if __name__ == "__main__":
    main()