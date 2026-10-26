
from bs4 import BeautifulSoup
import re


class UML2HTMLSemanticMetrics:

    def __init__(self, uml_metrics, html_metrics):
        self.uml_metrics = uml_metrics
        self.html_metrics = html_metrics

    def uml_2_html_semantics(self, uml_filename, html_filename):       

        with open(html_filename, encoding="utf-8") as f:
            html = BeautifulSoup(f, "html.parser")
        with open(uml_filename, encoding="utf-8") as f:
            uml = BeautifulSoup(f, "lxml-xml")

        uml_classes = uml.find_all("packagedElement", {"xmi:type": "uml:Class"})
        

        html_forms = html.find_all("form", attrs={"id" : re.compile(".")})     
        result = {}
        result["uml_filename"] = str(uml_filename)
        result["html_filename"] = str(html_filename)

        result[f"n_uml_2_html_class_to_form"] = 0
        result[f"n_uml_2_html_class_attributes_to_html"] = 0

        result["n_uml_2_html_primitive_attributes_to_html"] = 0
        result["n_uml_2_html_primitive_types_evaluable"] = 0
        result["n_uml_2_html_primitive_types_preserved"] = 0

        result["n_uml_2_html_enumeration_attributes_to_html"] = 0
        result["n_uml_2_html_enumeration_literals_to_html"] = 0
        result["n_uml_2_html_enumeration_cardinalities_evaluable"] = 0
        result["n_uml_2_html_enumeration_cardinalities_preserved"] = 0

        result["n_uml_2_html_associations_to_html"] = 0
        result["n_uml_2_html_association_ends_to_html"] = 0
        result["n_uml_2_html_associations_both_ends_to_html"] = 0

        result["n_uml_2_html_association_ends_with_label"] = 0
        result["n_uml_2_html_association_cardinalities_evaluable"] = 0
        result["n_uml_2_html_association_cardinalities_preserved"] = 0


        result["n_uml_2_html_attributes_with_label"] = 0
        result["n_uml_2_html_attributes_with_nonempty_label"] = 0


        for uml_class in uml_classes:
            class_name = uml_class.get('name','')
            html_form = self.find_html_form_for_class(
                html,
                class_name
            )
            if html_form is None:
                result[f"has_uml_2_html_class_{class_name}_to_form"] = False
            else:
                result[f"n_uml_2_html_class_to_form"] = result[f"n_uml_2_html_class_to_form"] + 1
                result[f"has_uml_2_html_class_{class_name}_to_form"] = True
                html_forms.remove(html_form) 
                

                
                result[f"uml_2_html_class_{class_name}_primitive_type_undefined"] = 0
                
                uml_class_attributes = uml_class.find_all("ownedAttribute", attrs={"xmi:type":"uml:Property"})
                
                for uml_class_attribute in uml_class_attributes:
                    uml_class_attribute_name = uml_class_attribute.get("name")    
                         
                    if self.uml_metrics.is_uml_class_attribute_primitive_type(
                        uml_class_attribute
                    ):



                        html_control = self.find_html_control_for_attribute(
                            html_form,
                            uml_class_attribute_name,
                            ["input", "textarea"]
                        )

                        input_primitive_type = (
                            self.uml_class_attribute_type_to_html_input_type(
                                uml_class_attribute
                            )
                        )
                        # Attribute presence
                        attribute_found = html_control is not None

                        result[
                            f"has_uml_2_html_class_{class_name}_attribute_"
                            f"{uml_class_attribute_name}"
                        ] = attribute_found

                        if attribute_found:
                            result[
                                "n_uml_2_html_class_attributes_to_html"
                            ] += 1

                            result[
                                "n_uml_2_html_primitive_attributes_to_html"
                            ] += 1




                        if input_primitive_type is not None:
                            # Expected primitive type
                            result[
                                f"uml_2_html_class_{class_name}_attribute_"
                                f"{uml_class_attribute_name}_primitive_type"
                            ] = input_primitive_type

                            # Actual HTML control type
                            html_control_type = None

                            if html_control is not None:

                                if html_control.name == "textarea":
                                    html_control_type = "text"

                                elif html_control.name == "input":
                                    html_control_type = html_control.get(
                                        "type",
                                        "text"
                                    ).lower()

                            # Type preservation
                            primitive_type_preserved = (
                                html_control is not None
                                and html_control_type == input_primitive_type
                            )

                            result[
                                f"has_uml_2_html_class_{class_name}_attribute_"
                                f"{uml_class_attribute_name}_primitive_type"
                            ] = primitive_type_preserved

                            if html_control is not None:
                                result[
                                    "n_uml_2_html_primitive_types_evaluable"
                                ] += 1

                                if primitive_type_preserved:
                                    result[
                                        "n_uml_2_html_primitive_types_preserved"
                                    ] += 1

                        else:
                            result[
                                f"uml_2_html_class_{class_name}_primitive_type_undefined"
                            ] += 1



                        # Label associated with the matched control
                        html_label = None

                        if (
                            html_control is not None
                            and html_control.get("id")
                        ):
                            html_label = html_form.find(
                                "label",
                                attrs={
                                    "for": html_control.get("id")
                                }
                            )

                        result[
                            f"has_uml_2_html_class_{class_name}_attribute_"
                            f"{uml_class_attribute_name}_label"
                        ] = html_label is not None

                        html_label_text = (
                            html_label.get_text(" ", strip=True)
                            if html_label is not None
                            else ""
                        )

                        result[
                            f"has_uml_2_html_class_{class_name}_attribute_"
                            f"{uml_class_attribute_name}_label_text"
                        ] = bool(html_label_text)

                        if html_label is not None:
                            result[
                                "n_uml_2_html_attributes_with_label"
                            ] += 1

                        if html_label_text:
                            result[
                                "n_uml_2_html_attributes_with_nonempty_label"
                            ] += 1

                                  
                    # Enumeration
                    elif uml_class_attribute.get("type") is not None:                 
                        class_attribute_type_id = uml_class_attribute.get("type")
                        if class_attribute_type_id is not None:
                            class_attribute_type = uml.find("packagedElement", {"xmi:id": class_attribute_type_id, "xmi:type":"uml:Enumeration"})   
                            if class_attribute_type is not None:
                                tagname = "select"
                                uml_enumeration_name = class_attribute_type.get('name')

                                html_select = self.find_html_control_for_attribute(
                                    html_form,
                                    uml_class_attribute_name,
                                    ["select"]
                                )


                                
                                if html_select is not None:


                                    result[
                                        f"uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_enumeration_control_type"
                                    ] = "select"

                                    html_label = None

                                    if html_select.get("id"):
                                        html_label = html_form.find(
                                            "label",
                                            attrs={
                                                "for": html_select.get("id")
                                            }
                                        )

                                    result[
                                        f"has_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_label"
                                    ] = html_label is not None

                                    html_label_text = (
                                        html_label.get_text(" ", strip=True)
                                        if html_label is not None
                                        else ""
                                    )

                                    result[
                                        f"has_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_label_text"
                                    ] = bool(html_label_text)

                                    if html_label is not None:
                                        result[
                                            "n_uml_2_html_attributes_with_label"
                                        ] += 1

                                    if html_label_text:
                                        result[
                                            "n_uml_2_html_attributes_with_nonempty_label"
                                        ] += 1

                                    result[
                                        f"has_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}"
                                    ] = True


                                    result[
                                        "n_uml_2_html_class_attributes_to_html"
                                    ] += 1
                                    result[
                                        "n_uml_2_html_enumeration_attributes_to_html"
                                    ] += 1

                                    result[f"uml_2_html_class_{class_name}_attribute_{uml_class_attribute_name}_{tagname}_enumeration_type"] = uml_enumeration_name


                                    enumeration_literals = class_attribute_type.find_all("ownedLiteral") 
                                    
                                    for enumeration_literal in enumeration_literals:
                                        enumeration_literal_name = enumeration_literal.get('name')
                                        
                                        html_option = None

                                        for option in html_select.find_all("option"):
                                            option_value = self.normalize_identifier(
                                                option.get("value")
                                            )

                                            option_text = self.normalize_identifier(
                                                option.get_text(" ", strip=True)
                                            )

                                            expected_literal = self.normalize_identifier(
                                                enumeration_literal_name
                                            )

                                            if (
                                                option_value == expected_literal
                                                or option_text == expected_literal
                                            ):
                                                html_option = option
                                                break

                                        result[f"has_uml_2_html_class_{class_name}_attribute_{uml_class_attribute_name}_{tagname}_enumeration_type_{uml_enumeration_name}_literal_option_{enumeration_literal_name}"] = html_option is not None

                                        if html_option is not None:
                                            result[
                                                "n_uml_2_html_enumeration_literals_to_html"
                                            ] += 1

                                        
                                        html_option_text = html_option.text if html_option else None                                




                                        
                                        result[f"has_uml_2_html_class_{class_name}_attribute_{uml_class_attribute_name}_{tagname}_enumeration_type_{uml_enumeration_name}_literal_option_{enumeration_literal_name}_text"] =  (
                                            html_option is not None
                                            and self.normalize_identifier(
                                                html_option.get_text(" ", strip=True)
                                            )
                                            == self.normalize_identifier(
                                                enumeration_literal_name
                                            )
                                        ) if html_option_text else False



                                    cardinality_metrics = (
                                        self.evaluate_enumeration_cardinality(
                                            uml_class_attribute,
                                            "select",
                                            html_select=html_select
                                        )
                                    )

                                    if cardinality_metrics["cardinality"] is not None:
                                        result[
                                            "n_uml_2_html_enumeration_cardinalities_evaluable"
                                        ] += 1

                                        if cardinality_metrics["cardinality"]:
                                            result[
                                                "n_uml_2_html_enumeration_cardinalities_preserved"
                                            ] += 1







                                    result[
                                        f"has_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_cardinality_single_multiple"
                                    ] = cardinality_metrics["single_multiple"]

                                    result[
                                        f"has_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_cardinality_optional_mandatory"
                                    ] = cardinality_metrics["optional_mandatory"]

                                    result[
                                        f"has_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_cardinality"
                                    ] = cardinality_metrics["cardinality"]


                                    result[
                                        f"html_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_multiple"
                                    ] = cardinality_metrics["html_multiple"]

                                    result[
                                        f"html_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_required"
                                    ] = cardinality_metrics["html_required"]

                                    result[
                                        f"html_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_empty_option"
                                    ] = cardinality_metrics["html_empty_option"]

                                    result[
                                        f"has_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_{tagname}_enumeration_type_"
                                        f"{uml_enumeration_name}_cardinality"
                                    ] = cardinality_metrics["cardinality"]

                                else:
                                    tagname = "input"
                                    uml_enumeration_name = class_attribute_type.get("name")
                                    enumeration_literals = class_attribute_type.find_all(
                                        "ownedLiteral"
                                    )

                                    html_enumeration_inputs = (
                                        self.find_html_enumeration_inputs_for_attribute(
                                            html_form,
                                            uml_class_attribute_name
                                        )
                                    )

                                    html_enumeration_input_found = bool(
                                        html_enumeration_inputs
                                    )

                                    # Attribute presence
                                    result[
                                        f"has_uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}"
                                    ] = html_enumeration_input_found

                                    if html_enumeration_input_found:
                                        result[
                                            "n_uml_2_html_class_attributes_to_html"
                                        ] += 1

                                        result[
                                            "n_uml_2_html_enumeration_attributes_to_html"
                                        ] += 1


                                    # Type of HTML representation selected by the LLM
                                    html_enumeration_input_types = {
                                        control.get("type", "").lower()
                                        for control in html_enumeration_inputs
                                    }

                                    enumeration_control_type = (
                                        next(iter(html_enumeration_input_types))
                                        if len(html_enumeration_input_types) == 1
                                        else "mixed"
                                        if html_enumeration_input_types
                                        else None
                                    )

                                    result[
                                        f"uml_2_html_class_{class_name}_attribute_"
                                        f"{uml_class_attribute_name}_enumeration_control_type"
                                    ] = enumeration_control_type


                                    if enumeration_control_type is not None:

                                        cardinality_metrics = (
                                            self.evaluate_enumeration_cardinality(
                                                uml_class_attribute,
                                                enumeration_control_type,
                                                html_inputs=html_enumeration_inputs
                                            )
                                        )


                                        if cardinality_metrics["cardinality"] is not None:
                                            result[
                                                "n_uml_2_html_enumeration_cardinalities_evaluable"
                                            ] += 1

                                            if cardinality_metrics["cardinality"]:
                                                result[
                                                    "n_uml_2_html_enumeration_cardinalities_preserved"
                                                ] += 1






                                        result[
                                            f"has_uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_cardinality_single_multiple"
                                        ] = cardinality_metrics["single_multiple"]

                                        result[
                                            f"has_uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_cardinality_optional_mandatory"
                                        ] = cardinality_metrics["optional_mandatory"]

                                        result[
                                            f"has_uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_cardinality"
                                        ] = cardinality_metrics["cardinality"]

                                        result[
                                            f"html_uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_required"
                                        ] = cardinality_metrics["html_required"]

                                    else:
                                        result[
                                            f"has_uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_cardinality_single_multiple"
                                        ] = None

                                        result[
                                            f"has_uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_cardinality_optional_mandatory"
                                        ] = None

                                        result[
                                            f"has_uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_cardinality"
                                        ] = None



                                    for enumeration_literal in enumeration_literals:
                                        enumeration_literal_name = enumeration_literal.get(
                                            "name"
                                        )

                                        expected_literal = self.normalize_identifier(
                                            enumeration_literal_name
                                        )

                                        html_enumeration_input = None



                                        html_enumeration_label = None

                                        for control in html_enumeration_inputs:
                                            control_value = self.normalize_identifier(
                                                control.get("value")
                                            )

                                            control_label = None

                                            if control.get("id"):
                                                control_label = html_form.find(
                                                    "label",
                                                    attrs={
                                                        "for": control.get("id")
                                                    }
                                                )

                                            control_label_text = (
                                                self.normalize_identifier(
                                                    control_label.get_text(
                                                        " ",
                                                        strip=True
                                                    )
                                                )
                                                if control_label is not None
                                                else ""
                                            )

                                            if (
                                                control_value == expected_literal
                                                or control_label_text == expected_literal
                                            ):
                                                html_enumeration_input = control
                                                html_enumeration_label = control_label
                                                break

                                        result[
                                            f"has_uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_{tagname}_enumeration_type_"
                                            f"{uml_enumeration_name}_literal_option_"
                                            f"{enumeration_literal_name}"
                                        ] = html_enumeration_input is not None


                                        if html_enumeration_input is not None:
                                            result[
                                                "n_uml_2_html_enumeration_literals_to_html"
                                            ] += 1

                                        html_enumeration_label_text = (
                                            html_enumeration_label.get_text(
                                                " ",
                                                strip=True
                                            )
                                            if html_enumeration_label is not None
                                            else ""
                                        )

                                        result[
                                            f"has_uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_{tagname}_enumeration_type_"
                                            f"{uml_enumeration_name}_literal_option_"
                                            f"{enumeration_literal_name}_label"
                                        ] = bool(html_enumeration_label_text)

                                    if html_enumeration_input_found:
                                        result[
                                            f"uml_2_html_class_{class_name}_attribute_"
                                            f"{uml_class_attribute_name}_{tagname}_enumeration_type"
                                        ] = uml_enumeration_name

        associations = uml.find_all(
            "packagedElement",
            {"xmi:type": "uml:Association"}
        )

        for association_index, association in enumerate(
            associations
        ):

            association_ends = association.find_all(
                "ownedEnd",
                recursive=False
            )

            if len(association_ends) != 2:
                raise ValueError(
                    "Only binary UML associations are supported "
                    f"by the current evaluator. "
                    f"Association index: {association_index}"
                )

            navigable_owned_end_ids = set(
                association.get(
                    "navigableOwnedEnd",
                    ""
                ).split()
            )

            association_end_ids = {
                association_end.get("xmi:id")
                for association_end in association_ends
            }

            if navigable_owned_end_ids != association_end_ids:
                raise ValueError(
                    "All association ends must be explicitly "
                    "navigable in the experimental corpus. "
                    f"Association index: {association_index}"
                )




            association_end_results = []

            for end_index, association_end in enumerate(
                association_ends
            ):

                opposite_end = association_ends[
                    1 - end_index
                ]

                association_end_name = association_end.get(
                    "name"
                )

                target_class_id = association_end.get(
                    "type"
                )

                owner_class_id = opposite_end.get(
                    "type"
                )

                target_class = uml.find(
                    "packagedElement",
                    {
                        "xmi:id": target_class_id,
                        "xmi:type": "uml:Class"
                    }
                )

                owner_class = uml.find(
                    "packagedElement",
                    {
                        "xmi:id": owner_class_id,
                        "xmi:type": "uml:Class"
                    }
                )

                if (
                    target_class is None
                    or owner_class is None
                    or association_end_name is None
                ):
                    raise ValueError(
                        "Invalid UML association end at "
                        f"association {association_index}, "
                        f"end {end_index}"
                    )

                target_class_name = target_class.get(
                    "name"
                )

                owner_class_name = owner_class.get(
                    "name"
                )

                html_form = self.find_html_form_for_class(
                    html,
                    owner_class_name
                )

                representation = {
                    "control_type": None,
                    "control": None,
                    "inputs": []
                }

                if html_form is not None:
                    representation = (
                        self.find_html_association_end_representation(
                            html_form,
                            association_end_name
                        )
                    )

                association_end_found = (
                    representation["control_type"]
                    is not None
                )

                association_end_results.append(
                    association_end_found
                )

                base_key = (
                    f"uml_2_html_association_"
                    f"{association_index}_"
                    f"{owner_class_name}_"
                    f"{association_end_name}_"
                    f"{target_class_name}"
                )

                result[
                    f"has_{base_key}"
                ] = association_end_found

                result[
                    f"{base_key}_control_type"
                ] = representation["control_type"]

                if association_end_found:
                    result[
                        "n_uml_2_html_association_ends_to_html"
                    ] += 1

                    association_end_label = None

                    if representation["control"] is not None:
                        control_id = representation[
                            "control"
                        ].get("id")

                        if control_id:
                            association_end_label = html_form.find(
                                "label",
                                attrs={
                                    "for": control_id
                                }
                            )


                    association_end_input_labels = []

                    for control in representation["inputs"]:
                        control_id = control.get("id")

                        if not control_id:
                            continue

                        label = html_form.find(
                            "label",
                            attrs={
                                "for": control_id
                            }
                        )

                        if label is not None:
                            association_end_input_labels.append(
                                label
                            )


                    association_end_has_label = (
                        association_end_label is not None
                        or bool(association_end_input_labels)
                    )

                    result[
                        f"has_{base_key}_label"
                    ] = association_end_has_label


                    if association_end_has_label:
                        result[
                            "n_uml_2_html_association_ends_with_label"
                        ] += 1


                    cardinality_metrics = (
                        self.evaluate_association_end_cardinality(
                            association_end,
                            representation["control_type"],
                            html_control=representation[
                                "control"
                            ],
                            html_inputs=representation[
                                "inputs"
                            ]
                        )
                    )


                    if cardinality_metrics["cardinality"] is not None:
                        result[
                            "n_uml_2_html_association_cardinalities_evaluable"
                        ] += 1

                        if cardinality_metrics["cardinality"]:
                            result[
                                "n_uml_2_html_association_cardinalities_preserved"
                            ] += 1




                    result[
                        f"has_{base_key}_"
                        f"cardinality_single_multiple"
                    ] = cardinality_metrics[
                        "single_multiple"
                    ]

                    result[
                        f"has_{base_key}_"
                        f"cardinality_optional_mandatory"
                    ] = cardinality_metrics[
                        "optional_mandatory"
                    ]

                    result[
                        f"has_{base_key}_cardinality"
                    ] = cardinality_metrics[
                        "cardinality"
                    ]

                    result[
                        f"html_{base_key}_multiple"
                    ] = cardinality_metrics[
                        "html_multiple"
                    ]

                    result[
                        f"html_{base_key}_required"
                    ] = cardinality_metrics[
                        "html_required"
                    ]

                else:

                    result[
                        f"has_{base_key}_label"
                    ] = False

                    result[
                        f"html_{base_key}_multiple"
                    ] = None

                    result[
                        f"html_{base_key}_required"
                    ] = None


                    result[
                        f"has_{base_key}_"
                        f"cardinality_single_multiple"
                    ] = None

                    result[
                        f"has_{base_key}_"
                        f"cardinality_optional_mandatory"
                    ] = None

                    result[
                        f"has_{base_key}_cardinality"
                    ] = None

            association_found = any(
                association_end_results
            )

            both_ends_found = all(
                association_end_results
            )

            result[
                f"has_uml_2_html_association_"
                f"{association_index}"
            ] = association_found

            result[
                f"has_uml_2_html_association_"
                f"{association_index}_both_ends"
            ] = both_ends_found

            if association_found:
                result[
                    "n_uml_2_html_associations_to_html"
                ] += 1

            if both_ends_found:
                result[
                    "n_uml_2_html_associations_both_ends_to_html"
                ] += 1

        return result

    def uml_class_attribute_type_to_html_input_type(self, uml_class_attribute):
        uml_class_attribute_name = uml_class_attribute.get("name","")
        if re.search(r"password", uml_class_attribute_name, flags=re.IGNORECASE):
            return 'password'
        if re.search(r"email", uml_class_attribute_name, flags=re.IGNORECASE):
            return 'email'    
        uml_class_attribute_primitive_type_name = self.uml_metrics.get_uml_class_attribute_primitive_type_name(uml_class_attribute)
        # print("UML class attribute primitive type",uml_class_attribute_primitive_type_name)    
        html_input_type = self.uml_primitive_type_to_html_input_type(uml_class_attribute_primitive_type_name)
        # print("HTML input type:",html_input_type)
        return html_input_type


    def uml_primitive_type_to_html_input_type(self, uml_type):
        if uml_type == "EString":
            return "text"
        elif uml_type == "EInteger" \
            or uml_type == "EIntegerObject" \
            or uml_type == "EFloat" \
            or uml_type == "EFloatObject" \
            or uml_type == "EDouble" \
            or uml_type == "ELong" \
            or uml_type == "EShort" \
            or uml_type == "EByte" \
            or uml_type == "EInt":
            return "number"
        elif uml_type == "EBoolean":
            return "checkbox"
        elif uml_type == "EDate":
            return "date"

        return None




    def evaluate_enumeration_cardinality(
        self,
        uml_attribute,
        control_type,
        html_select=None,
        html_inputs=None
    ):
        """
        Evaluate preservation of UML enumeration cardinality in HTML.

        Returns:
            {
                "single_multiple": True | False | None,
                "optional_mandatory": True | False | None,
                "cardinality": True | False | None,
                "html_multiple": True | False | None,
                "html_required": True | False | None,
                "html_empty_option": True | False | None,
            }
        """

        html_inputs = html_inputs or []

        uml_is_single = self.uml_metrics.is_uml_attribute_single(
            uml_attribute
        )
        uml_is_multiple = self.uml_metrics.is_uml_attribute_multiple(
            uml_attribute
        )
        uml_is_mandatory = self.uml_metrics.is_uml_attribute_mandatory(
            uml_attribute
        )

        single_multiple_preserved = None
        optional_mandatory_preserved = None

        html_multiple = None
        html_required = None
        html_empty_option = None

        if control_type == "select" and html_select is not None:

            html_multiple = html_select.has_attr("multiple")
            html_required = html_select.has_attr("required")

            html_empty_option = (
                html_select.find(
                    "option",
                    attrs={"value": ""}
                ) is not None
            )

            # Upper bound: single vs multiple
            single_multiple_preserved = (
                (uml_is_multiple and html_multiple)
                or (uml_is_single and not html_multiple)
            )

            # Lower bound: optional vs mandatory
            optional_mandatory_preserved = (
                html_required
                if uml_is_mandatory
                else not html_required
            )

        elif control_type == "radio":

            # A coherent radio group represents a single value.
            normalized_names = {
                self.normalize_identifier(
                    control.get("name")
                )
                for control in html_inputs
                if self.normalize_identifier(
                    control.get("name")
                )
            }

            is_radio_group = (
                len(html_inputs) > 0
                and len(normalized_names) == 1
            )

            single_multiple_preserved = (
                uml_is_single and is_radio_group
            )

            html_required = any(
                control.has_attr("required")
                for control in html_inputs
            )

            optional_mandatory_preserved = (
                html_required
                if uml_is_mandatory
                else not html_required
            )

        elif control_type == "checkbox":

            # Independent checkboxes naturally support multiple values.
            single_multiple_preserved = uml_is_multiple

            html_required = any(
                control.has_attr("required")
                for control in html_inputs
            )

            if self.uml_metrics.is_uml_attribute_multiple_optional(
                uml_attribute
            ):
                optional_mandatory_preserved = (
                    not html_required
                )

            elif self.uml_metrics.is_uml_attribute_multiple_mandatory(
                uml_attribute
            ):
                # Static HTML checkbox markup does not let this evaluator
                # establish "at least one selected" reliably.
                optional_mandatory_preserved = None

            else:
                # Checkbox representation is incompatible with a
                # single-valued UML enumeration.
                optional_mandatory_preserved = (
                    html_required
                    if uml_is_mandatory
                    else not html_required
                )

        elif control_type == "mixed":

            # Mixed radio/checkbox representation does not provide
            # an unambiguous cardinality mapping.
            single_multiple_preserved = None
            optional_mandatory_preserved = None

        if (
            single_multiple_preserved is None
            or optional_mandatory_preserved is None
        ):
            cardinality_preserved = None
        else:
            cardinality_preserved = (
                single_multiple_preserved
                and optional_mandatory_preserved
            )

        return {
            "single_multiple": single_multiple_preserved,
            "optional_mandatory": optional_mandatory_preserved,
            "cardinality": cardinality_preserved,
            "html_multiple": html_multiple,
            "html_required": html_required,
            "html_empty_option": html_empty_option,
        }



    def normalize_identifier(self, value):
        if value is None:
            return ""

        return re.sub(
            r"[^a-zA-Z0-9]",
            "",
            str(value)
        ).lower()
    
    def identifier_matches(self, value, expected_name):
        normalized_value = self.normalize_identifier(value)
        normalized_expected = self.normalize_identifier(expected_name)

        if not normalized_value or not normalized_expected:
            return False

        return normalized_expected in normalized_value

    def find_html_form_for_class(self, html, class_name):
        forms = html.find_all("form")

        for form in forms:
            if self.identifier_matches(
                form.get("id"),
                class_name
            ):
                return form

        return None

    def find_html_control_for_attribute(
        self,
        html_form,
        uml_attribute_name,
        tag_names
    ):
        controls = html_form.find_all(tag_names)

        # First preference: exact normalized HTML name.
        for control in controls:
            if (
                self.normalize_identifier(control.get("name"))
                == self.normalize_identifier(uml_attribute_name)
            ):
                return control

        # Second preference: HTML id.
        for control in controls:
            if self.identifier_matches(
                control.get("id"),
                uml_attribute_name
            ):
                return control

        # Third preference: associated label.
        for control in controls:
            control_id = control.get("id")

            if not control_id:
                continue

            label = html_form.find(
                "label",
                attrs={"for": control_id}
            )

            if (
                label is not None
                and self.identifier_matches(
                    label.get_text(" ", strip=True),
                    uml_attribute_name
                )
            ):
                return control

        return None

    def find_html_enumeration_inputs_for_attribute(
        self,
        html_form,
        uml_attribute_name
    ):
        candidate_controls = []

        for control in html_form.find_all("input"):
            input_type = control.get(
                "type",
                ""
            ).lower()

            if input_type in (
                "radio",
                "checkbox"
            ):
                candidate_controls.append(control)

        # First preference: exact normalized HTML name.
        controls_by_name = [
            control
            for control in candidate_controls
            if (
                self.normalize_identifier(
                    control.get("name")
                )
                == self.normalize_identifier(
                    uml_attribute_name
                )
            )
        ]

        if controls_by_name:
            return controls_by_name

        # Second preference: HTML id.
        return [
            control
            for control in candidate_controls
            if self.identifier_matches(
                control.get("id"),
                uml_attribute_name
            )
        ]


    def find_html_association_end_representation(
        self,
        html_form,
        association_end_name
    ):
        """
        Find the HTML representation selected by the LLM
        for a navigable UML association end.

        No specific HTML widget is required.
        """

        # 1. Select
        html_select = self.find_html_control_for_attribute(
            html_form,
            association_end_name,
            ["select"]
        )

        if html_select is not None:
            return {
                "control_type": "select",
                "control": html_select,
                "inputs": []
            }

        # 2. Radio / checkbox group
        html_inputs = (
            self.find_html_enumeration_inputs_for_attribute(
                html_form,
                association_end_name
            )
        )

        if html_inputs:
            input_types = {
                control.get("type", "").lower()
                for control in html_inputs
            }

            control_type = (
                next(iter(input_types))
                if len(input_types) == 1
                else "mixed"
            )

            return {
                "control_type": control_type,
                "control": None,
                "inputs": html_inputs
            }

        # 3. Scalar input / textarea
        html_control = self.find_html_control_for_attribute(
            html_form,
            association_end_name,
            ["input", "textarea"]
        )

        if html_control is not None:

            if html_control.name == "input":
                input_type = html_control.get(
                    "type",
                    "text"
                ).lower()

                # These elements do not represent an association value.
                if input_type in (
                    "submit",
                    "button",
                    "reset"
                ):
                    html_control = None

        if html_control is not None:
            return {
                "control_type": html_control.name,
                "control": html_control,
                "inputs": []
            }

        return {
            "control_type": None,
            "control": None,
            "inputs": []
        }


    def evaluate_association_end_cardinality(
        self,
        association_end,
        control_type,
        html_control=None,
        html_inputs=None
    ):
        html_inputs = html_inputs or []

        uml_is_single = (
            self.uml_metrics.is_uml_association_end_single(
                association_end
            )
        )

        uml_is_multiple = (
            self.uml_metrics.is_uml_association_end_multiple(
                association_end
            )
        )

        uml_is_mandatory = (
            self.uml_metrics.is_uml_association_end_mandatory(
                association_end
            )
        )

        single_multiple_preserved = None
        optional_mandatory_preserved = None

        html_multiple = None
        html_required = None

        if control_type == "select" and html_control is not None:

            html_multiple = html_control.has_attr(
                "multiple"
            )

            html_required = html_control.has_attr(
                "required"
            )

            single_multiple_preserved = (
                (uml_is_multiple and html_multiple)
                or
                (uml_is_single and not html_multiple)
            )

            optional_mandatory_preserved = (
                html_required
                if uml_is_mandatory
                else not html_required
            )

        elif control_type == "radio":

            normalized_names = {
                self.normalize_identifier(
                    control.get("name")
                )
                for control in html_inputs
                if self.normalize_identifier(
                    control.get("name")
                )
            }

            is_radio_group = (
                len(html_inputs) > 0
                and len(normalized_names) == 1
            )

            single_multiple_preserved = (
                uml_is_single
                and is_radio_group
            )

            html_required = any(
                control.has_attr("required")
                for control in html_inputs
            )

            optional_mandatory_preserved = (
                html_required
                if uml_is_mandatory
                else not html_required
            )

        elif control_type == "checkbox":

            single_multiple_preserved = (
                uml_is_multiple
            )

            html_required = any(
                control.has_attr("required")
                for control in html_inputs
            )

            if (
                self.uml_metrics
                .is_uml_association_end_multiple_optional(
                    association_end
                )
            ):
                optional_mandatory_preserved = (
                    not html_required
                )

            elif (
                self.uml_metrics
                .is_uml_association_end_multiple_mandatory(
                    association_end
                )
            ):
                # Static checkbox markup does not allow us
                # to establish "at least one" reliably.
                optional_mandatory_preserved = None

            else:
                optional_mandatory_preserved = (
                    html_required
                    if uml_is_mandatory
                    else not html_required
                )

        elif control_type in (
            "input",
            "textarea"
        ) and html_control is not None:

            # One scalar control represents one association value.
            single_multiple_preserved = (
                True
                if uml_is_single
                else None
            )

            if (
                html_control.name == "input"
                and html_control.get(
                    "type",
                    "text"
                ).lower() == "hidden"
            ):
                # required has no useful validation semantics
                # for hidden inputs.
                html_required = None
                optional_mandatory_preserved = None

            else:
                html_required = html_control.has_attr(
                    "required"
                )

                optional_mandatory_preserved = (
                    html_required
                    if uml_is_mandatory
                    else not html_required
                )

        elif control_type == "mixed":

            single_multiple_preserved = None
            optional_mandatory_preserved = None

        if (
            single_multiple_preserved is None
            or optional_mandatory_preserved is None
        ):
            cardinality_preserved = None
        else:
            cardinality_preserved = (
                single_multiple_preserved
                and optional_mandatory_preserved
            )

        return {
            "single_multiple": single_multiple_preserved,
            "optional_mandatory": optional_mandatory_preserved,
            "cardinality": cardinality_preserved,
            "html_multiple": html_multiple,
            "html_required": html_required,
        }

        