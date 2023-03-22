def create_model(args, model_name):
    model_name_ = str(model_name).lower()
    if model_name_ == "efficientnet".lower():
        from flamby.datasets.fed_isic2019.model import Baseline

        model = Baseline()
    else:
        raise ValueError(f"Model {model_name} not implemented")

    return model
