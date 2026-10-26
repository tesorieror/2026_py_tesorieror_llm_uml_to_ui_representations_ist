from bs4 import BeautifulSoup

class UMLMetrics:

    def generate_uml_metrics(self, uml_filename):
        with open(uml_filename, encoding="utf-8") as f:
            uml = BeautifulSoup(f, "lxml-xml")
        result = {}
        result["uml_filename"] = str(uml_filename)
        uml_classes = uml.find_all("packagedElement", {"xmi:type": "uml:Class"})
        result["n_uml_classes"] = uml_classes is not None and len(uml_classes) or 0
        uml_enumerations = uml.find_all("packagedElement", {"xmi:type": "uml:Enumeration"})
        result[f"n_uml_enumerations"] =  uml_enumerations is not None and len(uml_enumerations) or 0
        associations = uml.find_all("packagedElement", {"xmi:type": "uml:Association"})
        result['n_uml_associations'] = len(associations)
        attributes = uml.find_all("ownedAttribute")
        result['n_uml_attributes'] = len(attributes)
        result["n_uml_primitive_attributes"] = 0
        result["n_uml_enumeration_attributes"] = 0
        result["n_uml_enumeration_attribute_literals"] = 0
        literals = uml.find_all("ownedLiteral")
        result['n_uml_enumeration_literals'] = len(literals)
        ends = uml.find_all("ownedEnd")
        result['n_uml_ends'] = len(ends)

        # Classes and attributes
        for uml_class in uml_classes:
            class_name = uml_class.get("name")
            uml_class_attributes = uml_class.find_all("ownedAttribute")        
            result[f"n_uml_{class_name}_attributes"] = len(uml_class_attributes) is not None and len(uml_class_attributes) or 0

            for uml_attribute in uml_class_attributes:
                attribute_name = uml_attribute.get("name")
                attribute_type_id = uml_attribute.get("type")


                result[f"uml_{class_name}_attribute_{attribute_name}_cardinality_type"] = self.uml_attribute_cardinality_type(uml_attribute)

                result[f"has_uml_{class_name}_attribute_{attribute_name}_cardinality_Single"] =  self.is_uml_attribute_single(uml_attribute)
                result[f"has_uml_{class_name}_attribute_{attribute_name}_cardinality_Multiple"] = self.is_uml_attribute_multiple(uml_attribute)
                result[f"has_uml_{class_name}_attribute_{attribute_name}_cardinality_Optional"] = self.is_uml_attribute_optional(uml_attribute)
                result[f"has_uml_{class_name}_attribute_{attribute_name}_cardinality_Mandatory"] = self.is_uml_attribute_mandatory(uml_attribute)

                result[f"has_uml_{class_name}_attribute_{attribute_name}_cardinality_Single_Optional"] =  self.is_uml_attribute_single_optional(uml_attribute)
                result[f"has_uml_{class_name}_attribute_{attribute_name}_cardinality_Single_Mandatory"] =  self.is_uml_attribute_single_mandatory(uml_attribute)
                result[f"has_uml_{class_name}_attribute_{attribute_name}_cardinality_Multiple_Optional"] = self.is_uml_attribute_multiple_optional(uml_attribute)
                result[f"has_uml_{class_name}_attribute_{attribute_name}_cardinality_Multiple_Mandatory"] = self.is_uml_attribute_multiple_mandatory(uml_attribute)


                if attribute_type_id is not None:
                    attribute_type = uml.find("packagedElement", {"xmi:id": attribute_type_id})                         
                    if attribute_type is not None and attribute_type.get("xmi:type") is not None and attribute_type.get("xmi:type") == "uml:Enumeration":                    
                        result["n_uml_enumeration_attributes"] += 1
                        enumeration_literals = attribute_type.find_all("ownedLiteral")
                        result["n_uml_enumeration_attribute_literals"] += len(
                            enumeration_literals
                        )
                        result[f"has_uml_{class_name}_attribute_{attribute_name}_type_enumeration"] = True
                        result[f"uml_{class_name}_attribute_{attribute_name}_type"] = attribute_type.get('name')
                    else:
                        result[f"has_uml_{class_name}_attribute_{attribute_name}_type_enumeration"] = False
                        result[f"uml_{class_name}_attribute_{attribute_name}_type"] = None
                else:
                    result[f"has_uml_{class_name}_attribute_{attribute_name}_type_enumeration"] = False
                    
                    has_uml_attribute_primitive_type = self.is_uml_class_attribute_primitive_type(uml_attribute)
                    result[f"has_uml_{class_name}_attribute_{attribute_name}_type_primitive"] = has_uml_attribute_primitive_type

                    if has_uml_attribute_primitive_type:     
                        result["n_uml_primitive_attributes"] += 1           
                        uml_attribute_primitive_type_name = self.get_uml_class_attribute_primitive_type_name(uml_attribute)
                        result[f"uml_{class_name}_attribute_{attribute_name}_type"] = uml_attribute_primitive_type_name
                    else:
                        result[f"uml_{class_name}_attribute_{attribute_name}_type"] = None


        # Enumerations

        for uml_enumeration in uml_enumerations:
            enumeration_name = uml_enumeration.get("name")
            enumeration_literals = uml_enumeration.find_all("ownedLiteral")
            result[f"n_{enumeration_name}_enumeration_literals"] = len(enumeration_literals) is not None and len(enumeration_literals) or 0
            for uml_enumeration_literal in enumeration_literals:
                literal_name = uml_enumeration_literal.get("name")
                result[f"has_uml_enumeration_{enumeration_name}_literal_{literal_name}"] = True
            
        # Associations
        
        for association_index, association in enumerate(associations):
            association_name = association.get("name")
            association_ends = association.find_all("ownedEnd")
            result[f"n_{association_name}_{association_index}_association_ends"] = len(association_ends) is not None and len(association_ends) or 0

            print("Association type:", self.uml_association_cardinality(association))

            result[f"uml_association_{association_name}_{association_index}_cardinality_type"] = self.uml_association_cardinality(association)
            result[f"has_uml_association_{association_name}_{association_index}_cardinality_One_to_One"] = self.is_uml_association_one_to_one(association)
            result[f"has_uml_association_{association_name}_{association_index}_cardinality_One_to_Many"] = self.is_uml_association_one_to_many(association)
            result[f"has_uml_association_{association_name}_{association_index}_cardinality_Many_to_Many"] = self.is_uml_association_many_to_many(association)
            i = 0
            for uml_association_end in association_ends:
                end_name = uml_association_end.get("name") 
                end_type_id = uml_association_end.get("type")
                end_type_name = "undefined"
                if end_type_id is not None:
                    end_type = uml.find("packagedElement", {"xmi:id": end_type_id})    
                    end_type_name = end_type.get("name","undefined")                
                location = "source" if i==0 else "target" if i==1 else "association"
                result[f"uml_association_{association_name}_{association_index}_{location}_end_{end_name}_type"] = end_type_name

                result[f"uml_association_{association_name}_{association_index}_{location}_end_{end_name}_cardinality_type"] = self.uml_association_end_cardinality_type(uml_association_end)
                result[f"has_uml_association_{association_name}_{association_index}_{location}_end_{end_name}_cardinality_Single"] = self.is_uml_association_end_single(uml_association_end)
                result[f"has_uml_association_{association_name}_{association_index}_{location}_end_{end_name}_cardinality_Multiple"] = self.is_uml_association_end_multiple(uml_association_end)
                result[f"has_uml_association_{association_name}_{association_index}_{location}_end_{end_name}_cardinality_Optional"] = self.is_uml_association_end_optional(uml_association_end)
                result[f"has_uml_association_{association_name}_{association_index}_{location}_end_{end_name}_cardinality_Mandatory"] = self.is_uml_association_end_mandatory(uml_association_end)
                result[f"has_uml_association_{association_name}_{association_index}_{location}_end_{end_name}_cardinality_Single_Optional"] = self.is_uml_association_end_single_optional(uml_association_end)
                result[f"has_uml_association_{association_name}_{association_index}_{location}_end_{end_name}_cardinality_Single_Mandatory"] = self.is_uml_association_end_single_mandatory(uml_association_end)
                result[f"has_uml_association_{association_name}_{association_index}_{location}_end_{end_name}_cardinality_Multiple_Optional"] = self.is_uml_association_end_multiple_optional(uml_association_end)
                result[f"has_uml_association_{association_name}_{association_index}_{location}_end_{end_name}_cardinality_Multiple_Mandatory"] = self.is_uml_association_end_multiple_mandatory(uml_association_end)
                i = i + 1
        return result

    def is_uml_attribute_optional(self, uml_attribute):
        attribute_cardinality_value = self.uml_attribute_cardinality(uml_attribute)
        return attribute_cardinality_value in ["0_1", "0_*"]

    def is_uml_attribute_mandatory(self, uml_attribute):
        attribute_cardinality_value = self.uml_attribute_cardinality(uml_attribute)
        return attribute_cardinality_value in ["1_1", "1_*"]

    def is_uml_attribute_single(self, uml_attribute):
        attribute_cardinality_value = self.uml_attribute_cardinality(uml_attribute)
        return attribute_cardinality_value in ["1_1", "0_1"]    

    def is_uml_attribute_multiple(self, uml_attribute):
        attribute_cardinality_value = self.uml_attribute_cardinality(uml_attribute)
        return attribute_cardinality_value in ["0_*", "1_*"]

    def is_uml_attribute_single_optional(self, uml_attribute):
        attribute_cardinality_value = self.uml_attribute_cardinality(uml_attribute)
        return attribute_cardinality_value == "0_1"

    def is_uml_attribute_single_mandatory(self, uml_attribute):
        attribute_cardinality_value = self.uml_attribute_cardinality(uml_attribute)
        return attribute_cardinality_value == "1_1"

    def is_uml_attribute_multiple_optional(self, uml_attribute):
        attribute_cardinality_value = self.uml_attribute_cardinality(uml_attribute)
        return attribute_cardinality_value == "0_*"

    def is_uml_attribute_multiple_mandatory(self, uml_attribute):
        attribute_cardinality_value = self.uml_attribute_cardinality(uml_attribute)
        return attribute_cardinality_value == "1_*"

    def uml_attribute_cardinality_type(self, uml_attribute):
        attribute_cardinality_value = self.uml_attribute_cardinality(uml_attribute)
        if attribute_cardinality_value == "1_1":
            return "Single_Mandatory"
        elif attribute_cardinality_value == "0_1":
            return "Single_Optional"
        elif attribute_cardinality_value == "0_*":
            return "Multiple_Optional" 
        elif attribute_cardinality_value == "1_*":
            return "Multiple_Mandatory"
        else:
            return "Multiple_Mandatory"

    def uml_attribute_cardinality(self, uml_attribute):
        uml_attribute_lower_multiplicity = uml_attribute.find("lowerValue") 
        if uml_attribute_lower_multiplicity is not None:
            uml_attribute_lower_multiplicity_value = uml_attribute_lower_multiplicity.get("value")
            lower_multiplicity =  uml_attribute_lower_multiplicity_value if uml_attribute_lower_multiplicity_value is not None else "0"
        else:
            lower_multiplicity = "1"

        uml_attribute_upper_multiplicity = uml_attribute.find("upperValue")
        if uml_attribute_upper_multiplicity is not None:
            uml_attribute_upper_multiplicity_value = uml_attribute_upper_multiplicity.get("value")
            upper_multiplicity = uml_attribute_upper_multiplicity_value if uml_attribute_upper_multiplicity_value is not None else "1"
        else:
            upper_multiplicity = "1"

        return f"{lower_multiplicity}_{upper_multiplicity}"

    def is_uml_association_one_to_one(self, association):
        return self.uml_association_cardinality(association) == "One_to_One"

    def is_uml_association_one_to_many(self, association):
        return self.uml_association_cardinality(association) == "One_to_Many"

    def is_uml_association_many_to_many(self, association):
        return self.uml_association_cardinality(association) == "Many_to_Many"


    def uml_association_cardinality(self, association):
        association_ends = association.find_all("ownedEnd")
        if len(association_ends) == 2:

            sourceEnd = association_ends[0]
            targetEnd = association_ends[1]

            isSourceEndSingle = self.is_uml_association_end_single(sourceEnd)
            isTargetEndSingle = self.is_uml_association_end_single(targetEnd)
            isSourceEndMultiple = self.is_uml_association_end_multiple(sourceEnd)
            isTargetEndMultiple = self.is_uml_association_end_multiple(targetEnd)

            if(isSourceEndSingle and isTargetEndSingle):
                return "One_to_One"
            elif (isSourceEndSingle and isTargetEndMultiple) or (isSourceEndMultiple and isTargetEndSingle):
                return "One_to_Many"
            elif (isSourceEndMultiple and isTargetEndMultiple):
                return "Many_to_Many"
                        
        return "Many_to_Many"

    def is_uml_association_end_optional(self, association_end):
        association_end_cardinality_value = self.uml_association_end_cardinality(association_end)
        return association_end_cardinality_value in ["0_1", "0_*"]

    def is_uml_association_end_mandatory(self, association_end):
        association_end_cardinality_value = self.uml_association_end_cardinality(association_end)
        return association_end_cardinality_value in ["1_1", "1_*"]

    def is_uml_association_end_single(self, association_end):
        association_end_cardinality_value = self.uml_association_end_cardinality(association_end)
        return association_end_cardinality_value in ["1_1", "0_1"]  

    def is_uml_association_end_multiple(self, association_end):
        association_end_cardinality_value = self.uml_association_end_cardinality(association_end)
        return association_end_cardinality_value in ["0_*", "1_*"]

    def is_uml_association_end_single_optional(self, association_end):
        association_end_cardinality_value = self.uml_association_end_cardinality(association_end)
        return association_end_cardinality_value == "0_1"  

    def is_uml_association_end_single_mandatory(self, association_end):
        association_end_cardinality_value = self.uml_association_end_cardinality(association_end)
        return association_end_cardinality_value == "1_1"

    def is_uml_association_end_multiple_optional(self, association_end):
        association_end_cardinality_value = self.uml_association_end_cardinality(association_end)
        return association_end_cardinality_value == "0_*"

    def is_uml_association_end_multiple_mandatory(self, association_end):
        association_end_cardinality_value = self.uml_association_end_cardinality(association_end)
        return association_end_cardinality_value == "1_*"  
    
    def uml_association_end_cardinality_type(self, association_end):
        association_end_cardinality_value = self.uml_association_end_cardinality(association_end)
        if association_end_cardinality_value == "1_1":
            return "Single_Mandatory"
        elif association_end_cardinality_value == "0_1":
            return "Single_Optional"
        elif association_end_cardinality_value == "0_*":
            return "Multiple_Optional" 
        elif association_end_cardinality_value == "1_*":
            return "Multiple_Mandatory"
        else:
            return "Multiple_Mandatory"

    def uml_association_end_cardinality(self, association_end):
        uml_association_end_lower_multiplicity = association_end.find("lowerValue") 
        if uml_association_end_lower_multiplicity is not None:
            uml_association_end_lower_multiplicity_value = uml_association_end_lower_multiplicity.get("value")
            lower_multiplicity =  uml_association_end_lower_multiplicity_value if uml_association_end_lower_multiplicity_value is not None else "0"
        else:
            lower_multiplicity = "1"

        uml_association_end_upper_multiplicity = association_end.find("upperValue")
        if uml_association_end_upper_multiplicity is not None:
            uml_association_end_upper_multiplicity_value = uml_association_end_upper_multiplicity.get("value")
            upper_multiplicity = uml_association_end_upper_multiplicity_value if uml_association_end_upper_multiplicity_value is not None else "1"
        else:
            upper_multiplicity = "1"

        return f"{lower_multiplicity}_{upper_multiplicity}"

    def is_uml_class_attribute_enumetarion_type(self, uml_class_attribute):
        uml = list(uml_class_attribute.parents)[-1]
        class_attribute_type_id = uml_class_attribute.get("type")
        if class_attribute_type_id is not None:
            class_attribute_type = uml.find("packagedElement", {"xmi:id": class_attribute_type_id})   
            return class_attribute_type is not None \
                and class_attribute_type.get("xmi:type") is not None \
                and class_attribute_type.get("xmi:type") == "uml:Enumeration"
        return False

    def is_uml_class_attribute_primitive_type(self, uml_attribute):
        class_attribute_type = uml_attribute.find("type")
        return class_attribute_type is not None \
            and class_attribute_type.get("xmi:type") is not None \
                and class_attribute_type.get("xmi:type") == "uml:PrimitiveType"  

    def get_uml_class_attribute_primitive_type_name(self, uml_attribute):
        return self.uml_class_attribute_primitive_type(uml_attribute) is not None and self.uml_class_attribute_primitive_type(uml_attribute) or "Undefined"

    def uml_class_attribute_primitive_type(self, uml_attribute):
        attribute_type = uml_attribute.find("type")    
        if(attribute_type.get("href")):
            attribute_type_name = attribute_type.get("href").split("#")[-1]
            return attribute_type_name
        return None