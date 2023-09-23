import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

def load_params_from_tf(py_model:nn.Module, tf_model:list):
    """
    Load and update the parameters from TensorFlow.js to PyTorch nn.Module.

    Args:
        py_model (nn.Module): A PyTorch neural network structure.
        tf_model (list): A list read from a JSON file containing metadata for the TensorFlow.js model.

    Returns:
        nn.Module: An updated PyTorch neural network structure.

    Raises:
        Exception: If certain layer structures do not align between PyTorch and TensorFlow.js.
        KeyError: If a model layer is not aligned.

    This function loads and updates the parameters from a TensorFlow.js model to a PyTorch nn.Module.
    It compares layer names between the two models and assigns the TensorFlow.js parameters to the
    corresponding layers in the PyTorch model.

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
    """
    Process a PyTorch state dictionary to convert it into a Python dictionary.

    Args:
        state_dict (dict): A PyTorch state dictionary containing model parameters.

    Returns:
        dict: A Python dictionary where keys are parameter names and values are
              NumPy arrays representing the parameter values.

    This function takes a PyTorch state dictionary, which typically contains the
    parameters of a neural network model, and converts it into a Python dictionary.
    Each key in the resulting dictionary corresponds to a parameter's name, and the
    corresponding value is a NumPy array containing the parameter's values.

    """
    lr_py = {}
    for key, value in state_dict.items():
        lr_py[key] = value.cpu().detach().numpy().tolist()
    return lr_py


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a logistic regression model.

        Args:
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.

        """
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the logistic regression model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the sigmoid function.

        """
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class CNN_WEB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x