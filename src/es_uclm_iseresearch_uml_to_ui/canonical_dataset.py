import json
from itertools import product
from pathlib import Path

import pandas as pd

from es_uclm_iseresearch_uml_to_ui.config import Config


class CanonicalDatasetBuilder:
    ID_COLUMNS = [
        "model",
        "uml_model",
        "representation",
        "iteration",
    ]

    UML_COLUMNS = [
        "n_uml_classes",
        "n_uml_enumerations",
        "n_uml_associations",
        "n_uml_attributes",
        "n_uml_primitive_attributes",
        "n_uml_enumeration_attributes",
        "n_uml_enumeration_attribute_literals",
        "n_uml_enumeration_literals",
        "n_uml_ends",
    ]

    HTML_COLUMNS = [
        "is_html_filename",
        "n_html_forms",
        "n_html_buttons",
        "n_html_inputs",
        "n_html_input_button",
        "n_html_input_submit",
        "n_html_input_checkbox",
        "n_html_input_date",
        "n_html_input_datetime_local",
        "n_html_input_email",
        "n_html_input_password",
        "n_html_input_number",
        "n_html_input_reset",
        "n_html_input_radio",
        "n_html_input_range",
        "n_html_input_text",
        "n_html_label",
        "n_html_select",
        "n_html_option",
        "n_html_table",
    ]

    SEMANTIC_COLUMNS = [
        "n_uml_2_html_class_to_form",
        "n_uml_2_html_class_attributes_to_html",
        "n_uml_2_html_primitive_attributes_to_html",
        "n_uml_2_html_primitive_types_evaluable",
        "n_uml_2_html_primitive_types_preserved",
        "n_uml_2_html_enumeration_attributes_to_html",
        "n_uml_2_html_enumeration_literals_to_html",
        "n_uml_2_html_enumeration_cardinalities_evaluable",
        "n_uml_2_html_enumeration_cardinalities_preserved",
        "n_uml_2_html_associations_to_html",
        "n_uml_2_html_association_ends_to_html",
        "n_uml_2_html_associations_both_ends_to_html",
        "n_uml_2_html_association_ends_with_label",
        "n_uml_2_html_association_cardinalities_evaluable",
        "n_uml_2_html_association_cardinalities_preserved",
        "n_uml_2_html_attributes_with_label",
        "n_uml_2_html_attributes_with_nonempty_label",
    ]

    RATIO_COLUMNS = [
        "metric_class_coverage",
        "metric_attribute_coverage",
        "metric_primitive_attribute_coverage",
        "metric_primitive_type_evaluability",
        "metric_primitive_type_fidelity",
        "metric_enumeration_attribute_coverage",
        "metric_enumeration_literal_coverage",
        "metric_enumeration_cardinality_evaluability",
        "metric_enumeration_cardinality_fidelity",
        "metric_attribute_label_coverage",
        "metric_attribute_nonempty_label_coverage",
        "metric_association_coverage",
        "metric_association_end_coverage",
        "metric_association_both_ends_coverage",
        "metric_association_end_label_coverage",
        "metric_association_cardinality_evaluability",
        "metric_association_cardinality_fidelity",
    ]

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _load_json(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _require_fields(data, fields, source):
        missing = [
            field
            for field in fields
            if field not in data
        ]

        if missing:
            raise KeyError(
                f"Missing required fields in {source}: {missing}"
            )

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if pd.isna(numerator) or pd.isna(denominator):
            return float("nan")

        if denominator == 0:
            return float("nan")

        return numerator / denominator

    def _load_uml_metrics(self):
        metrics_by_model = {}

        for uml_model in self.config.uml_models:
            filename = (
                self.config.metrics_uml_json_filename(
                    uml_model
                )
            )

            if not Path(filename).is_file():
                raise FileNotFoundError(
                    f"UML metrics file not found: {filename}"
                )

            data = self._load_json(filename)

            self._require_fields(
                data,
                self.UML_COLUMNS,
                filename,
            )

            metrics_by_model[uml_model] = {
                field: data[field]
                for field in self.UML_COLUMNS
            }

        return metrics_by_model

    def _validate_generation_statistics(
        self,
        dataframe,
    ):
        required_columns = (
            self.ID_COLUMNS
            + [
                "n_html_blocks",
                "n_js_blocks",
                "n_css_blocks",
                "metric_html_generated",
                "metric_js_generated",
                "metric_css_generated",
                "metric_generation_success",
            ]
        )

        missing = [
            field
            for field in required_columns
            if field not in dataframe.columns
        ]

        if missing:
            raise KeyError(
                "Missing generation-statistics columns: "
                f"{missing}"
            )

        duplicated = dataframe.duplicated(
            subset=self.ID_COLUMNS,
            keep=False,
        )

        if duplicated.any():
            duplicated_ids = dataframe.loc[
                duplicated,
                self.ID_COLUMNS,
            ]

            raise ValueError(
                "Duplicate experimental attempt identifiers "
                "found:\n"
                + duplicated_ids.to_string(index=False)
            )

        expected_attempts = set(
            product(
                self.config.model_names,
                self.config.uml_models,
                self.config.representations,
                range(self.config.iterations),
            )
        )

        observed_attempts = set(
            dataframe[
                self.ID_COLUMNS
            ].itertuples(
                index=False,
                name=None,
            )
        )

        missing_attempts = (
            expected_attempts
            - observed_attempts
        )

        unexpected_attempts = (
            observed_attempts
            - expected_attempts
        )

        if (
            missing_attempts
            or unexpected_attempts
        ):
            raise ValueError(
                "Generation statistics do not match "
                "the experimental design. "
                f"Missing attempts: "
                f"{sorted(missing_attempts)}. "
                f"Unexpected attempts: "
                f"{sorted(unexpected_attempts)}."
            )

        generation_success_expected = (
            dataframe["n_html_blocks"] == 1
        )

        generation_success_observed = (
            dataframe[
                "metric_generation_success"
            ].astype(bool)
        )

        if not generation_success_expected.equals(
            generation_success_observed
        ):
            raise ValueError(
                "Invariant violation: "
                "metric_generation_success must be "
                "equivalent to (n_html_blocks == 1)."
            )

        block_invariants = [
            (
                "metric_html_generated",
                "n_html_blocks",
            ),
            (
                "metric_js_generated",
                "n_js_blocks",
            ),
            (
                "metric_css_generated",
                "n_css_blocks",
            ),
        ]

        for (
            metric_column,
            count_column,
        ) in block_invariants:
            expected = (
                dataframe[count_column] > 0
            )

            observed = dataframe[
                metric_column
            ].astype(bool)

            if not expected.equals(observed):
                raise ValueError(
                    f"Invariant violation: "
                    f"{metric_column} must be "
                    f"equivalent to "
                    f"({count_column} > 0)."
                )

    def _add_ratios(self, row):
        row["metric_class_coverage"] = (
            self._safe_ratio(
                row[
                    "n_uml_2_html_class_to_form"
                ],
                row["n_uml_classes"],
            )
        )

        row["metric_attribute_coverage"] = (
            self._safe_ratio(
                row[
                    "n_uml_2_html_"
                    "class_attributes_to_html"
                ],
                row["n_uml_attributes"],
            )
        )

        row[
            "metric_primitive_attribute_coverage"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "primitive_attributes_to_html"
            ],
            row["n_uml_primitive_attributes"],
        )

        row[
            "metric_primitive_type_evaluability"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "primitive_types_evaluable"
            ],
            row[
                "n_uml_2_html_"
                "primitive_attributes_to_html"
            ],
        )

        row[
            "metric_primitive_type_fidelity"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "primitive_types_preserved"
            ],
            row[
                "n_uml_2_html_"
                "primitive_types_evaluable"
            ],
        )

        row[
            "metric_enumeration_attribute_coverage"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "enumeration_attributes_to_html"
            ],
            row[
                "n_uml_enumeration_attributes"
            ],
        )

        row[
            "metric_enumeration_literal_coverage"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "enumeration_literals_to_html"
            ],
            row[
                "n_uml_enumeration_attribute_literals"
            ],
        )

        row[
            "metric_enumeration_"
            "cardinality_evaluability"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "enumeration_cardinalities_evaluable"
            ],
            row[
                "n_uml_2_html_"
                "enumeration_attributes_to_html"
            ],
        )

        row[
            "metric_enumeration_"
            "cardinality_fidelity"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "enumeration_cardinalities_preserved"
            ],
            row[
                "n_uml_2_html_"
                "enumeration_cardinalities_evaluable"
            ],
        )

        row[
            "metric_attribute_label_coverage"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "attributes_with_label"
            ],
            row[
                "n_uml_2_html_"
                "class_attributes_to_html"
            ],
        )

        row[
            "metric_attribute_"
            "nonempty_label_coverage"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "attributes_with_nonempty_label"
            ],
            row[
                "n_uml_2_html_"
                "class_attributes_to_html"
            ],
        )

        row[
            "metric_association_coverage"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "associations_to_html"
            ],
            row["n_uml_associations"],
        )

        row[
            "metric_association_end_coverage"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "association_ends_to_html"
            ],
            row["n_uml_ends"],
        )

        row[
            "metric_association_"
            "both_ends_coverage"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "associations_both_ends_to_html"
            ],
            row["n_uml_associations"],
        )

        row[
            "metric_association_"
            "end_label_coverage"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "association_ends_with_label"
            ],
            row[
                "n_uml_2_html_"
                "association_ends_to_html"
            ],
        )

        row[
            "metric_association_"
            "cardinality_evaluability"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "association_cardinalities_evaluable"
            ],
            row[
                "n_uml_2_html_"
                "association_ends_to_html"
            ],
        )

        row[
            "metric_association_"
            "cardinality_fidelity"
        ] = self._safe_ratio(
            row[
                "n_uml_2_html_"
                "association_cardinalities_preserved"
            ],
            row[
                "n_uml_2_html_"
                "association_cardinalities_evaluable"
            ],
        )

        return row

    def _validate_ratios(self, dataframe):
        for column in self.RATIO_COLUMNS:
            non_na = dataframe[
                column
            ].dropna()

            invalid = (
                (non_na < 0)
                | (non_na > 1)
            )

            if invalid.any():
                raise ValueError(
                    f"Ratio outside [0, 1] "
                    f"in {column}: "
                    f"{non_na[invalid].tolist()}"
                )

    def build(self):
        generation_filename = (
            self.config.generator_statistics_filename
        )

        if not Path(
            generation_filename
        ).is_file():
            raise FileNotFoundError(
                "Generation statistics file "
                "not found: "
                f"{generation_filename}"
            )

        generation_df = pd.read_csv(
            generation_filename
        )

        self._validate_generation_statistics(
            generation_df
        )

        uml_metrics_by_model = (
            self._load_uml_metrics()
        )

        rows = []

        for (
            _,
            generation_row,
        ) in generation_df.iterrows():
            row = generation_row.to_dict()

            model = row["model"]
            uml_model = row["uml_model"]
            representation = (
                row["representation"]
            )
            iteration = int(
                row["iteration"]
            )

            if (
                uml_model
                not in uml_metrics_by_model
            ):
                raise KeyError(
                    "No UML metrics loaded "
                    f"for model: {uml_model}"
                )

            row.update(
                uml_metrics_by_model[
                    uml_model
                ]
            )

            html_metrics_filename = (
                self.config
                .metrics_html_json_filename(
                    model,
                    uml_model,
                    representation,
                    iteration,
                )
            )

            semantic_metrics_filename = (
                self.config
                .metrics_semantics_json_filename(
                    model,
                    uml_model,
                    representation,
                    iteration,
                )
            )

            success = bool(
                row[
                    "metric_generation_success"
                ]
            )

            if success:
                if not Path(
                    html_metrics_filename
                ).is_file():
                    raise FileNotFoundError(
                        "Successful attempt has "
                        "no HTML metrics: "
                        f"{html_metrics_filename}"
                    )

                if not Path(
                    semantic_metrics_filename
                ).is_file():
                    raise FileNotFoundError(
                        "Successful attempt has "
                        "no semantic metrics: "
                        f"{semantic_metrics_filename}"
                    )

                html_metrics = (
                    self._load_json(
                        html_metrics_filename
                    )
                )

                semantic_metrics = (
                    self._load_json(
                        semantic_metrics_filename
                    )
                )

                self._require_fields(
                    html_metrics,
                    self.HTML_COLUMNS,
                    html_metrics_filename,
                )

                self._require_fields(
                    semantic_metrics,
                    self.SEMANTIC_COLUMNS,
                    semantic_metrics_filename,
                )

                row.update(
                    {
                        field: html_metrics[
                            field
                        ]
                        for field
                        in self.HTML_COLUMNS
                    }
                )

                row.update(
                    {
                        field: semantic_metrics[
                            field
                        ]
                        for field
                        in self.SEMANTIC_COLUMNS
                    }
                )

            else:
                if Path(
                    html_metrics_filename
                ).is_file():
                    raise ValueError(
                        "Failed generation has "
                        "stale HTML metrics: "
                        f"{html_metrics_filename}"
                    )

                if Path(
                    semantic_metrics_filename
                ).is_file():
                    raise ValueError(
                        "Failed generation has "
                        "stale semantic metrics: "
                        f"{semantic_metrics_filename}"
                    )

                for field in self.HTML_COLUMNS:
                    row[field] = float("nan")

                for field in (
                    self.SEMANTIC_COLUMNS
                ):
                    row[field] = float("nan")

            self._add_ratios(row)

            rows.append(row)

        dataframe = pd.DataFrame(rows)

        self._validate_ratios(dataframe)

        return dataframe

    def save_csv(self, filename=None):
        if filename is None:
            filename = (
                self.config.results_path
                / "canonical_dataset.csv"
            )

        dataframe = self.build()

        dataframe.to_csv(
            filename,
            index=False,
            encoding="utf-8",
        )

        print(
            "Canonical dataset generated: "
            f"{filename} "
            f"({len(dataframe)} rows)"
        )

        return dataframe


def main():
    Config.initialize(
        "experiment/metadata.json"
    )

    builder = CanonicalDatasetBuilder(
        Config.instance()
    )

    builder.save_csv()


if __name__ == "__main__":
    main()