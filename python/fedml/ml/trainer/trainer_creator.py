from .my_model_trainer_classification import ModelTrainerCLS
from .my_model_trainer_nwp import ModelTrainerNWP
from .my_model_trainer_tag_prediction import ModelTrainerTAGPred


def create_model_trainer(model, args):
    """
    Create and return an appropriate model trainer based on the dataset type.

    Args:
        model: The neural network model to be trained.
        args: A dictionary containing training configuration parameters, including the dataset type.

    Returns:
        ModelTrainer: An instance of a model trainer tailored to the dataset type.
    """
    if args.dataset == "stackoverflow_lr":
        model_trainer = ModelTrainerTAGPred(model, args)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        model_trainer = ModelTrainerNWP(model, args)
    else:  # Default model trainer is for classification problem
        model_trainer = ModelTrainerCLS(model, args)
    return model_trainer
