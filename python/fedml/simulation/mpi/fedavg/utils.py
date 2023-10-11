import os
import numpy as np
import torch

def transform_list_to_tensor(model_params_list):
    """
    Transform a dictionary of model parameters from a list format to PyTorch tensors.

    Args:
        model_params_list (dict): A dictionary containing model parameters in list format.

    Returns:
        dict: A dictionary containing model parameters as PyTorch tensors.
    """
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list

def transform_tensor_to_list(model_params):
    """
    Transform a dictionary of model parameters from PyTorch tensors to a list format.

    Args:
        model_params (dict): A dictionary containing model parameters as PyTorch tensors.

    Returns:
        dict: A dictionary containing model parameters in list format.
    """
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params

def post_complete_message_to_sweep_process(args):
    """
    Post a completion message to a sweep process.

    This function creates a named pipe and writes a completion message to it, along with the provided arguments.

    Args:
        args: An object containing configuration parameters.
    """
    pipe_path = "./tmp/fedml"
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))
