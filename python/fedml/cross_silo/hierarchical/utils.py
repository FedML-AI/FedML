import os
from collections import OrderedDict

from .trainer.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .trainer.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .trainer.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG


# ref: https://discuss.pytorch.org/t/failed-to-load-model-trained-by-ddp-for-inference/84841/2?u=amir_zsh
def convert_model_params_from_ddp(ddp_model_params):
    model_params = OrderedDict()
    for k, v in ddp_model_params.items():
        name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        model_params[name] = v
    return model_params


def convert_model_params_to_ddp(ddp_model_params):
    model_params = OrderedDict()
    for k, v in ddp_model_params.items():
        name = f"module.{k}"  # add 'module.' of DataParallel/DistributedDataParallel
        model_params[name] = v
    return model_params


def post_complete_message_to_sweep_process(args):
    pipe_path = "./tmp/fedml"
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))


def get_model_trainer(model, args):
    if args.dataset == "stackoverflow_lr":
        model_trainer = MyModelTrainerTAG(model, args)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        model_trainer = MyModelTrainerNWP(model, args)
    else:  # default model trainer is for classification problem
        model_trainer = MyModelTrainerCLS(model, args)
    return model_trainer
