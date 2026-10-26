import requests
import json

from es_uclm_iseresearch_uml_to_ui.config import Config

# models = ["llama3", "mistral", "qwen2"]
# uml_models = {"uml1":"", "uml2":"", "uml3":"", "uml4":"", "uml5":"", "uml6":"", "uml7":"", "uml8":"", "uml9":"", "uml10":""}
# representations = ["xmi", "json", "text"]

class Prompter:

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance



    def __init__(self):
        self.config = Config.instance()

    def run_prompt(self, model, prompt):
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model["name"],
            "prompt": prompt,
            "stream": model["stream"],
            "options": model["options"]
        }
        response = requests.post(url, json=data)
        return response.json()

    def load_intro(self):
        with open(self.config.prompt_path_intro_filename, "r", encoding="utf-8") as f:
            return f.read()
    
    def create_prompts(self, intro, model_contents):
        prompts = {}
        for representation, models in model_contents.items():
            prompts[representation] = {}
            for model_name, content in models.items():
                prompt = f"{intro}\n\n{content}"
                prompts[representation][model_name] = prompt
        return prompts

    def save_prompts(self, prompts):
        for representation, models in prompts.items():
            for model_name, prompt in models.items():
                with open(self.config.prompt_filename(model_name, representation), "w", encoding="utf-8") as f:
                    f.write(prompt)
    
    def load_prompts(self, uml_models, representations):
        prompts = {}
        for representation in representations:
            prompts[representation] = {}
            for uml_model in uml_models:
                with open(self.config.prompt_filename(uml_model, representation), "r", encoding="utf-8") as f:
                    prompt = f.read()            
                prompts[representation][uml_model] = prompt
        return prompts

    def generate_prompts(self, model_contents, intro):         
        prompts = self.create_prompts(intro, model_contents)
        self.save_prompts(prompts)


    def run_prompts(self, models, prompts, iterations):
        responses = {}
        for model in models:   
            model_name = model["name"]
            print(f"Running prompts for model {model_name}...")
            responses[model_name] = {}
            responses[model_name]["config"] = model
            for representation, uml_models in prompts.items():
                responses[model_name][representation] = {}
                for uml_model_name, prompt in uml_models.items():
                    responses[model_name][representation][uml_model_name] = []
                    for i in range(iterations):                
                        print(f"Running on {model_name} prompt {i} for {uml_model_name} in {representation} representation...")
                        response = self.run_prompt(model, prompt)
                        responses[model_name][representation][uml_model_name].append(response)
                        print(f"Received of {model_name} in response {i} for {uml_model_name} in {representation} representation.")
                        # print(f"Response: {response}")
                        print(f"Saving from {model_name} response {i} for {uml_model_name} in {representation} representation...")
                        with open(self.config.raw_response_filename(model_name, uml_model_name, representation, i), "w", encoding="utf-8") as f:
                            f.write(json.dumps(response, indent=4))
                        print(f"Saved from {model_name} response {i} for {uml_model_name} in {representation} representation.")
        return responses

    def load_responses(self, model_names,uml_model_names, representations, iterations):
        responses = {}
        for model_name in model_names:
            responses[model_name] = {}
            for representation in representations:
                responses[model_name][representation] = {}
                for uml_model_name in uml_model_names:
                    responses[model_name][representation][uml_model_name] = []
                    for i in range(iterations):                
                        with open(self.config.raw_response_filename(model_name, uml_model_name, representation, i), "r", encoding="utf-8") as f:
                            response = json.load(f)
                        responses[model_name][representation][uml_model_name].append(response)
        return responses
