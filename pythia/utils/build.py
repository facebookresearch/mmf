import importlib

MODEL_KEY_TO_CLASS_NAME_MAPPING = {
    'top_down_bottom_up_modified': 'VQAMultiModalModel',
    # TODO: top_down_bottom_up should be different
    'top_down_bottom_up': 'VQAMultiModalModel',
    'visdial_top_down_bottom_up': 'VisDialMultiModalModel'
}


def build_model(config):
    models_module_key = "pythia.models"
    models_module = importlib.import_module(models_module_key)

    model_name = MODEL_KEY_TO_CLASS_NAME_MAPPING[config['model']]
    model_class = getattr(models_module, model_name)
    model = model_class(config)

    if hasattr(model, 'build'):
        model.build()

    return model
