import importlib


def build_model(config):
    models_module_key = "pythia.models"
    models_module = importlib.import_module(models_module_key)

    model_name = config['model']
    model_class = getattr(models_module, model_name)

    return model_class(config)
