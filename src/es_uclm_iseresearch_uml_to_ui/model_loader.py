from es_uclm_iseresearch_uml_to_ui.config import Config

class ModelLoader:

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


    def __init__(self):
        self.config = Config.instance()
    
    def load_model_contents(self, uml_models, representations):    
        model_contents = {}
        for representation in representations:
            model_contents[representation] = {}
            for model in uml_models:
                with open(self.config.model_representation_filename(model, representation), "r", encoding="utf-8") as f:
                    model_contents[representation][model] = f.read()
        return model_contents