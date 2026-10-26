import json
from pathlib import Path

class Config:
    _instance = None
    
    @classmethod
    def initialize(cls, metadata_file: str):
        if cls._instance is not None:
            raise RuntimeError("Config already initialized")

        cls._instance = cls(metadata_file)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            raise RuntimeError(
                "Config not initialized. Call Config.initialize() first"
            )

        return cls._instance

    def __init__(self, metadata_file: str):
        self.metadata = self._load_metadata(metadata_file)
    
    @property
    def input_path(self)->Path:
        return Path(self.metadata["input_path"])
    
    @property
    def output_path(self)->Path:
        return Path(self.metadata["output_path"])

    @property
    def prompt_intro_filename(self)->Path:
        return Path(self.metadata["prompt_intro_filename"])
    
    @property
    def prompt_path(self)->Path:
        return Path(self.metadata["prompt_path"])
    
    @property
    def prompt_path_intro_filename(self)->Path:
        return self.input_path / self.prompt_intro_filename

    @property
    def raw_response_path(self)->Path:
        return Path(self.metadata["raw_response_path"])

    @property
    def models(self):
        return self.metadata["models"]
    
    @property
    def model_names(self):
        return [model["name"] for model in self.models]

    @property
    def iterations(self):
        return self.metadata["iterations"]
    
    @property
    def uml_models(self):
        return self.metadata["uml_models"] 
    
    @property
    def representations(self):
        return self.metadata["representations"]
    
    @property
    def extracted_code_path(self)->Path:
        return Path(self.metadata["extracted_code_path"])

    @property
    def results_path(self)->Path:
        return Path(self.metadata["results_path"])

    @property
    def generator_statistics_filename(self)->Path:
        return self.results_path / "generation_statistics.csv"

    def model_representation_filename(self, model_name: str, representation: str)->Path:
        return self.input_path / f"{representation}" / f"{model_name}.{representation}"

    def model_uml_filename(self, model_name: str)->Path:
        return self.input_path / "uml" / f"{model_name}.uml"

    def _load_metadata(self, file)->dict:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)

    def prompt_filename(self, model_name: str, representation: str)->Path:
        return self.prompt_path / f"{model_name}_{representation}.txt"    

    def raw_response_filename(self, model_name: str, uml_model_name: str, representation: str, iteration: int)->Path:
        safe_model_name = self.model_filename_safe_name(model_name)
        return self.raw_response_path / f"{safe_model_name}_{uml_model_name}_{representation}_{iteration}.txt"    

    def extracted_code_path_for_response(self, model_name: str, uml_model_name: str, representation: str, iteration: int)->Path:
        safe_model_name = self.model_filename_safe_name(model_name)
        return self.extracted_code_path / f"{safe_model_name}_{uml_model_name}_{representation}_{iteration}"

    def extracted_code_html_filename(self, model_name: str, uml_model_name: str, representation: str, iteration: int)->Path:

        return self.extracted_code_path_for_response(model_name, uml_model_name, representation, iteration) / "index.html"
    
    def extracted_code_css_filename(self, model_name: str, uml_model_name: str, representation: str, iteration: int)->Path:
        return self.extracted_code_path_for_response(model_name, uml_model_name, representation, iteration) / "style.css"
    
    def extracted_code_js_filename(self, model_name: str, uml_model_name: str, representation: str, iteration: int)->Path:
        return self.extracted_code_path_for_response(model_name, uml_model_name, representation, iteration) / "script.js"
    
    def metrics_html_json_filename(self, model_name: str, uml_model_name: str, representation: str, iteration: int)->Path:
        safe_model_name = self.model_filename_safe_name(model_name)
        return self.results_path / f"{representation}" / f"{safe_model_name}_{uml_model_name}_{representation}_{iteration}_html_metrics.json"
    
    def metrics_uml_json_filename(self, uml_model_name: str)->Path:
        return self.results_path / "uml"/ f"{uml_model_name}_uml_metrics.json"
        
    def metrics_semantics_json_filename(self, model_name: str, uml_model_name: str, representation: str, iteration: int)->Path:
        safe_model_name = self.model_filename_safe_name(model_name)
        return self.results_path / f"{representation}" / f"{safe_model_name}_{uml_model_name}_{representation}_{iteration}_semantic_metrics.json"

    def model_filename_safe_name(self, model_name: str) -> str:
        return model_name.replace(":", "_")
