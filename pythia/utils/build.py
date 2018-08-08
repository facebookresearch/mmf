import importlib

MODEL_KEY_TO_CLASS_NAME_MAPPING = {
    'top_down_bottom_up_modified': 'VQAMultiModalModel',
    # TODO: top_down_bottom_up should be different
    'top_down_bottom_up': 'VQAMultiModalModel'
}


def build_model(config):
    models_module_key = "pythia.models"
    models_module = importlib.import_module(models_module_key)

    model_name = MODEL_KEY_TO_CLASS_NAME_MAPPING[config['model']]
    print(model_name)
    model_class = getattr(models_module, model_name)

    return model_class(config)
