import pandas as pd
from es_uclm_iseresearch_uml_to_ui.code_extractor import CodeExtractor
from es_uclm_iseresearch_uml_to_ui.config import Config

class CodeGeneratorStatistics:

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.responses = None
        self.code_extractor = CodeExtractor()        
        self.config = Config.instance()

    def compute_response_statistics(self, responses):
        self.responses = responses
        response_statistics = []
        for model_name, representations in self.responses.items():
            for representation, uml_models in representations.items():
                for uml_model_name, response in uml_models.items():
                    # for i in range(iterations):
                    for i, response_row in enumerate(response):
                        reg = {}                    
                        reg['uml_model'] = uml_model_name
                        reg['representation'] = representation
                        reg['iteration'] = i
                        print(f"Computing statistics for response {i} for {uml_model_name} in {representation} representation...")
                        # print(response_row)                
                        reg['model'] = model_name
                        reg['model_description'] = response_row['model']
                        reg['created_at'] = response_row['created_at']
                        reg['total_duration'] = response_row['total_duration']
                        reg['load_duration'] = response_row['load_duration']
                        reg['prompt_eval_count'] = response_row['prompt_eval_count']
                        reg['prompt_eval_duration'] = response_row['prompt_eval_duration']
                        reg['eval_count'] = response_row['eval_count']
                        reg['eval_duration'] = response_row['eval_duration']


                        html_blocks = self.code_extractor.extract_html_from_response(
                            response_row["response"]
                        )

                        js_blocks = self.code_extractor.extract_js_from_response(
                            response_row["response"]
                        )

                        css_blocks = self.code_extractor.extract_css_from_response(
                            response_row["response"]
                        )

                        reg["n_html_blocks"] = len(html_blocks)
                        reg["n_js_blocks"] = len(js_blocks)
                        reg["n_css_blocks"] = len(css_blocks)

                        reg["metric_html_generated"] = len(html_blocks) > 0
                        reg["metric_js_generated"] = len(js_blocks) > 0
                        reg["metric_css_generated"] = len(css_blocks) > 0

                        reg["metric_generation_success"] = (
                            len(html_blocks) == 1
                        )
                        response_statistics.append(reg)
        return response_statistics
    
    def save_generation_statistics_csv(self, statistics):
        # Convertir a DataFrame
        response_statistics_df = pd.DataFrame(statistics)
        # Mostrar DataFrame
        print(response_statistics_df)
        # Guardar como CSV
        response_statistics_df.to_csv(self.config.generator_statistics_filename, index=False, encoding="utf-8")
        print("CSV generado correctamente")