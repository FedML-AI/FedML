def create_model(args, model_name):
    model_name_ = str(model_name).lower()
    if model_name_ == "LR".lower():
        from flamby.datasets.fed_heart_disease.model import Baseline

        model = Baseline(args.input_dim, args.output_dim)
    else:
        raise ValueError(f"Model {model_name} not implemented")

    return model
