import torch
import torch.nn as nn
import pandas as pd

def load_params_from_tf(py_model:nn.Module, tf_model:list):
    """
    Load and update the parameters from tensorflow.js to pytorch nn.Module

    Args:
        py_model: An nn.Moudule network structure from pytorch
        tf_module: A list read from JSON file which stored the meta data of tensorflow.js model 
                   (length is number of layers, and has two keys in each layer, 'model' and 'params' respectively)

    Returns:
        An updated nn.Module network structure

    Raises:
        Exception: Certain layer structure is not aligned
        KeyError: Model layer is not aligned
    """
    state_dict = py_model.state_dict()
    py_layers = list(state_dict.keys())
    tf_layers = [d['model']['name'] for d in tf_model]
    tf_params_dict = {d['model']['name'] : torch.tensor(pd.Series(d['params'])) for d in tf_model}
    py_nlayers = len(py_layers)
    tf_nlayers = len(tf_layers)
    if tf_nlayers == py_nlayers:
        try:
            for py_layer, tf_layer in zip(py_layers, tf_layers):
                layer_shape = state_dict[py_layer].shape
                params_in = tf_params_dict[tf_layer]
                params_in = torch.reshape(params_in, layer_shape)

                state_dict[py_layer] = params_in
            py_model.load_state_dict(state_dict)
            return py_model
        except:
            raise Exception(f"Sorry, model structure did not align in pytorch layer {py_layer}, and tensorflow.js layer {tf_layer}!")
    else:
        raise TypeError("The model structure of pytorch and tensorflow.js is not aligned! Cannot transfer parameters accordingly.")

def process_state_dict(state_dict):
    lr_py = {}
    for key, value in state_dict.items():
        lr_py[key] = value.cpu().detach().numpy().tolist()
    return lr_py