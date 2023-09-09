import os
import torch
import numpy as np

def transform_list_to_tensor(model_params_list):
    """
    Transform a dictionary of model parameters from a list of NumPy arrays to PyTorch tensors.

    Args:
        model_params_list (dict): Dictionary of model parameters represented as NumPy arrays.

    Returns:
        dict: Dictionary of model parameters with tensors as values.
    """
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list

def transform_tensor_to_list(model_params):
    """
    Transform a dictionary of model parameters from PyTorch tensors to lists.

    Args:
        model_params (dict): Dictionary of model parameters represented as PyTorch tensors.

    Returns:
        dict: Dictionary of model parameters with lists as values.
    """
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params

def post_complete_message_to_sweep_process(args):
    """
    Post a completion message to a named pipe for communication with another process.

    Args:
        args: Additional information or configuration to include in the message.

    Returns:
        None
    """
    pipe_path = "./tmp/fedml"
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("Training is finished! \n%s\n" % (str(args)))
