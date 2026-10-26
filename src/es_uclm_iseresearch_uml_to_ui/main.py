print("LLM Python request")
import re
import json
from pathlib import Path

from es_uclm_iseresearch_uml_to_ui.config import Config
from es_uclm_iseresearch_uml_to_ui.code_generator import CodeGenerator
from es_uclm_iseresearch_uml_to_ui.metrics.uml_metrics import UMLMetrics
from es_uclm_iseresearch_uml_to_ui.metrics.html_metrics import HTMLMetrics
from es_uclm_iseresearch_uml_to_ui.metrics.uml_to_html_semantics import UML2HTMLSemanticMetrics

METADATA_FILENAME = "experiment/metadata.json"
# METADATA_FILENAME = "experiment/metadata_determinism_pilot.json"
Config.initialize(METADATA_FILENAME)

def build_form_id_class_attrs(class_name): 
    def match_attrs(tag):        
        pattern = re.compile(class_name, re.IGNORECASE)
        if tag.name != "form":
            return False

        # id
        if pattern.search(tag.get("id", "")):
            return True

        # class
        classes = " ".join(tag.get("class", []))
        if pattern.search(classes):
            return True

        # data-*
        for attr, value in tag.attrs.items():
            if attr.startswith("data-"):
                if isinstance(value, str) and pattern.search(value):
                    return True

        return False
    return match_attrs

def evaluate_code():
    config = Config.instance()
    model_names = [item["name"] for item in config.models]
    iterations = config.iterations
    uml_models = config.uml_models
    representations = config.representations

    html_metrics = HTMLMetrics()
    uml_metrics = UMLMetrics()
    uml_2_html_semantic_metrics = UML2HTMLSemanticMetrics(uml_metrics, html_metrics)

    for uml_model_name in uml_models:
        uml_filename = Config.instance().model_uml_filename(uml_model_name)
        uml_metrics_result = uml_metrics.generate_uml_metrics(uml_filename)
        uml_metrics_result_json = json.dumps(uml_metrics_result, indent=2, ensure_ascii=False)
        uml_metrics_filename = config.metrics_uml_json_filename(uml_model_name)
        # print("------------------------------")
        print(f"UML Metrics for {uml_filename}:")
        # print(uml_metrics_result_json)
        # print("------------------------------")
        with open(uml_metrics_filename,"w", encoding="utf-8") as f:
            f.write(uml_metrics_result_json)

    for model_name in model_names:
        for representation in representations:
            for uml_model_name in uml_models:
                for i in range(iterations):
                    html_filename = config.extracted_code_html_filename(model_name,uml_model_name, representation,i)
                    print(f"Evaluating extracted code for LLM {model_name} response {i} for {uml_model_name} in {representation} representation...")                   
                    if Path(html_filename).is_file():
                        html_filename = config.extracted_code_html_filename(model_name,uml_model_name, representation,i)
                        html_metrics_result = html_metrics.generate_html_metrics(html_filename)
                        html_metrics_result_json = json.dumps(html_metrics_result, indent=2, ensure_ascii=False)
                        html_metrics_filename = config.metrics_html_json_filename(model_name,uml_model_name, representation, i)
                        # print("------------------------------")
                        print(f"HTML Metrics for {html_filename}:")
                        # print(html_metrics_result_json)
                        # print("------------------------------")
                        with open(html_metrics_filename,"w", encoding="utf-8") as f:
                            f.write(html_metrics_result_json)
                        uml_filename = Config.instance().model_uml_filename(uml_model_name)
                        uml_2_html_semantics_result = uml_2_html_semantic_metrics.uml_2_html_semantics(uml_filename, html_filename)
                        uml_2_html_semantics_result_json = json.dumps(uml_2_html_semantics_result, indent=2, ensure_ascii=False)
                        uml_2_html_semantics_filename = config.metrics_semantics_json_filename(model_name,uml_model_name,representation,i)
                        # print("------------------------------")
                        print(f"UML2HTML Metrics for {uml_filename} and {html_filename}:")
                        # print(uml_2_html_semantics_result_json)
                        # print("------------------------------")
                        with open(uml_2_html_semantics_filename,"w", encoding="utf-8") as f:
                            f.write(uml_2_html_semantics_result_json)

def main():
    code_generator = CodeGenerator()
    code_generator.generate_code()
    evaluate_code()
   

if __name__ == "__main__":
    main()