from pythia.core.registry import Registry


def build_model(config):
    model_name = config['model']

    model_class = Registry.get_model_class(model_name)

    if model_class is None:
        Registry.get('writer').write("No model registered for name: %s" %
                                     model_name)
    model = model_class(config)

    if hasattr(model, 'build'):
        model.build()

    return model
