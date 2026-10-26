import re
from pathlib import Path
from es_uclm_iseresearch_uml_to_ui.config import Config
import shutil

class CodeExtractor:
    
    _instance = None
    
    @staticmethod
    def instance():
        if CodeExtractor._instance is None:
            CodeExtractor._instance = CodeExtractor()
        return CodeExtractor._instance

    def extract_html_from_response(self, response):
        return re.findall(r"```html\s*(.*?)```", response, re.DOTALL)

    def extract_js_from_response(self, response):
        return re.findall(r"```javascript\s*(.*?)```", response, re.DOTALL)

    def extract_css_from_response(self, response):
        return re.findall(r"```css\s*(.*?)```", response, re.DOTALL)

    def extract_code(self, responses):
        config = Config.instance()
        code = {}
        for model_name, representations in responses.items():
            code[model_name] = {}
            for representation, uml_models in representations.items():
                code[model_name][representation] = {}
                for uml_model_name, response in uml_models.items():
                    code[model_name][representation][uml_model_name] = {}
                    # for i in range(iterations):
                    for i, response_row in enumerate(response):
                        code[model_name][representation][uml_model_name][i] = {}
                        # Process the response as needed (e.g., extract relevant information, convert formats, etc.)
                        print(f"Processing response {i} for {uml_model_name} in {representation} representation...")
                        # Example: Extracting a specific field from the response
                        # response = responses[model_name][representation][uml_model_name][i]
                        response_content = response_row['response']
                        # print(response_content)
                        extracted_code_path_for_response = (
                            config.extracted_code_path_for_response(
                                model_name,
                                uml_model_name,
                                representation,
                                i
                            )
                        )

                        if extracted_code_path_for_response.exists():
                            shutil.rmtree(extracted_code_path_for_response)

                        extracted_code_path_for_response.mkdir(
                            parents=True,
                            exist_ok=True
                        )
                        # Path(f"experiment/extracted_code/{model_name}_{uml_model_name}_{representation}_{i}/src").mkdir(exist_ok=True)

                        response_html_content = self.extract_html_from_response(
                            response_content
                        )

                        html_content = ""

                        if len(response_html_content) == 1:
                            html_filename = config.extracted_code_html_filename(
                                model_name,
                                uml_model_name,
                                representation,
                                i
                            )

                            html_content = (
                                "<!-- Extracted from response 0 -->\n"
                                + response_html_content[0]
                            )

                            with open(
                                html_filename,
                                "w",
                                encoding="utf-8"
                            ) as f:
                                f.write(html_content)

                        elif len(response_html_content) == 0:
                            print(
                                f"No HTML block found in response {i} "
                                f"for {uml_model_name} in "
                                f"{representation} representation."
                            )

                        else:
                            print(
                                f"Multiple HTML blocks "
                                f"({len(response_html_content)}) found "
                                f"in response {i} for {uml_model_name} "
                                f"in {representation} representation. "
                                f"HTML will not be extracted."
                            )
                        
                        response_js_content = self.extract_js_from_response(response_content)
                        
                        js_content = ""

                        for j, js_content_j in enumerate(response_js_content):
                            # filename_regex = r"// (.*)\.js\n\n"
                            # filename = re.search(filename_regex, js_content, re.DOTALL).group(1)                    
                            js_content = js_content + f"\n /* Extracted from response {j}*/ \n" + js_content_j

                        js_filename = config.extracted_code_js_filename(model_name, uml_model_name, representation, i)
                        with open(js_filename, "w", encoding="utf-8") as f:
                            f.write(js_content)

                        response_css_content = self.extract_css_from_response(response_content)

                        css_content = ""
                        for j, css_content_j in enumerate(response_css_content):
                            # filename_regex = r"// (.*)\.css\n\n"
                            # filename = re.search(filename_regex, css_content, re.DOTALL).group(1)
                            css_content = css_content + f"\n /* Extracted from response {j}*/ \n" + css_content_j

                        css_filename = config.extracted_code_css_filename(model_name, uml_model_name, representation, i)
                        with open(css_filename, "w", encoding="utf-8") as f:
                            f.write(css_content)
                        # print(f"Extracted content from response {i} for {uml_model_name} in {representation} representation: {response_content}")
                        
                        code[model_name][representation][uml_model_name][i]["html"] = html_content
                        code[model_name][representation][uml_model_name][i]["css"] = css_content
                        code[model_name][representation][uml_model_name][i]["js"] = js_content
        return code