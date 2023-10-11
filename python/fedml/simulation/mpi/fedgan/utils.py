import os

import numpy as np
import torch


def transform_list_to_tensor(model_params_list):
    """
    Convert a dictionary of model parameters from NumPy arrays to PyTorch tensors.

    Args:
        model_params_list (dict): A dictionary containing model parameters.

    Returns:
        dict: A dictionary with model parameters converted to PyTorch tensors.
    """
    for net in model_params_list.keys():
        for k in model_params_list[net].keys():
            model_params_list[net][k] = torch.from_numpy(
                np.asarray(model_params_list[net][k])
            ).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    """
    Convert a dictionary of model parameters from PyTorch tensors to NumPy arrays.

    Args:
        model_params (dict): A dictionary containing model parameters as PyTorch tensors.

    Returns:
        dict: A dictionary with model parameters converted to NumPy arrays.
    """
    for net in model_params.keys():
        for k in model_params[net].keys():
            model_params[net][k] = model_params[net][k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    """
    Post a completion message to a named pipe.

    Args:
        args: Information or data to be included in the completion message.

    Returns:
        None
    """
    pipe_path = "./tmp/fedml"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("Training is finished! \n%s\n" % (str(args)))
