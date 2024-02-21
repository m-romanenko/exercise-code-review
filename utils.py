def print_model_summary(models):
    for model in models:
        print("Model summary:", model.name)


def calculate_parameter_count(model):
    return sum([layer.units for layer in model.layers if hasattr(layer, "units")])
