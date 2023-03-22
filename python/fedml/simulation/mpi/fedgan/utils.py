import os

import numpy as np
import torch


def transform_list_to_tensor(model_params_list):
    for net in model_params_list.keys():
        for k in model_params_list[net].keys():
            model_params_list[net][k] = torch.from_numpy(
                np.asarray(model_params_list[net][k])
            ).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for net in model_params.keys():
        for k in model_params[net].keys():
            model_params[net][k] = model_params[net][k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    pipe_path = "./tmp/fedml"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))
