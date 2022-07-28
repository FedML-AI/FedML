def create_model(args, model_name):
    model_name_ = str(model_name).lower()
    if model_name_ == "VNet".lower():
        from flamby.datasets.fed_lidc_idri.model import Baseline

        model = Baseline()
    else:
        raise ValueError(f"Model {model_name} not implemented")

    return model
