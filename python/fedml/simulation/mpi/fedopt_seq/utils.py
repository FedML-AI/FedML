import torch
import numpy as np
import os

def transform_list_to_tensor(model_params_list):
    """
    Convert a dictionary of model parameters from NumPy arrays in a list to PyTorch tensors.

    Args:
        model_params_list (dict): A dictionary of model parameters, where values are lists of NumPy arrays.

    Returns:
        dict: A dictionary of model parameters with values as PyTorch tensors.
    """
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list

def transform_tensor_to_list(model_params):
    """
    Convert a dictionary of model parameters from PyTorch tensors to lists of NumPy arrays.

    Args:
        model_params (dict): A dictionary of model parameters, where values are PyTorch tensors.

    Returns:
        dict: A dictionary of model parameters with values as lists of NumPy arrays.
    """
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params

def post_complete_message_to_sweep_process(args):
    """
    Send a completion message to a sweep process using a named pipe.

    Args:
        args (str): A string containing information about the training completion status or other relevant details.
    """
    pipe_path = "./tmp/fedml"
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))
