from .my_model_trainer_classification import ModelTrainerCLS
from .my_model_trainer_nwp import ModelTrainerNWP
from .my_model_trainer_tag_prediction import ModelTrainerTAGPred


def create_model_trainer(model, args):
    if args.dataset == "stackoverflow_lr":
        model_trainer = ModelTrainerTAGPred(model, args)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        model_trainer = ModelTrainerNWP(model, args)
    else:  # default model trainer is for classification problem
        model_trainer = ModelTrainerCLS(model, args)
    return model_trainer
