from es_uclm_iseresearch_uml_to_ui.config import Config
from es_uclm_iseresearch_uml_to_ui.model_loader import ModelLoader
from es_uclm_iseresearch_uml_to_ui.prompter import Prompter
from es_uclm_iseresearch_uml_to_ui.code_extractor import CodeExtractor
from es_uclm_iseresearch_uml_to_ui.code_generator_statistics import CodeGeneratorStatistics

from es_uclm_iseresearch_uml_to_ui.experiment_manifest import (
    save_manifest,
)

class CodeGenerator:
    
    _instance=None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.config = Config.instance()
        self.prompter = Prompter.instance()
        self.code_extractor = CodeExtractor.instance()
        self.code_generation_statistics = CodeGeneratorStatistics.instance()

    def generate_code(self):        
        models = self.config.models
        model_names = [item["name"] for item in models]
        iterations = self.config.iterations
        uml_models = self.config.uml_models
        representations = self.config.representations

        self.prompter = Prompter.instance()

        intro = self.prompter.load_intro()
        # print("Loaded metadata:", metadata)
        # print("Intro:", intro)
        # print("Load UML contents", load_model_contents(uml_models, representations))

        # Generate prompts and save them to files

        model_contents = ModelLoader.instance().load_model_contents(uml_models, representations)

        self.prompter.generate_prompts(model_contents, intro)

        print("Loading prompts...")
        prompts = self.prompter.load_prompts(uml_models, representations)
        # print(prompts)
        print("Loaded prompts.")


        save_manifest(
            "experiment/metadata.json"
        )

        ##############################################
        ##############################################
        # Uncomment the following lines to run the prompts and extract code, and comment out the rest of the code in main.py
        ##############################################
        ##############################################

        # Run prompts and save responses
        # self.prompter.run_prompts(models, prompts, iterations)





        print("Loading responses...")
        responses = self.prompter.load_responses(model_names, uml_models, representations, iterations)
        print("Loaded responses.")
        self.code_extractor.extract_code(responses)
        # response_statistics = compute_response_statistics(code)
        response_statistics = self.code_generation_statistics.compute_response_statistics(responses)
        self.code_generation_statistics.save_generation_statistics_csv(response_statistics)
