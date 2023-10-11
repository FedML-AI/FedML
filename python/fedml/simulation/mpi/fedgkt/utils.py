import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_state_dict(file):
    """
    Load a PyTorch state dictionary from a file.

    Args:
        file (str): The path to the file containing the state dictionary.

    Returns:
        dict: The loaded state dictionary.
    """
    try:
        pretrain_state_dict = torch.load(file)
    except AssertionError:
        pretrain_state_dict = torch.load(
            file, map_location=lambda storage, location: storage
        )
    return pretrain_state_dict

def get_flat_params_from(model):
    """
    Get a flat tensor containing all the parameters of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        torch.Tensor: A 1D tensor containing the flattened parameters.
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    """
    Set the parameters of a PyTorch model using a flat tensor of parameters.

    Args:
        model (nn.Module): The PyTorch model.
        flat_params (torch.Tensor): A 1D tensor containing the flattened parameters.
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind : prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size



class RunningAverage:
    """
    A simple class that maintains the running average of a quantity

    Example:
        ```
        loss_avg = RunningAverage()
        loss_avg.update(2)
        loss_avg.update(4)
        loss_avg() = 3.0
        ```

    Attributes:
        steps (int): The number of updates made to the running average.
        total (float): The cumulative sum of values for the running average.
    """

    def __init__(self):
        """
        Initialize a RunningAverage object.
        """
        self.steps = 0
        self.total = 0

    def update(self, val):
        """Update the running average with a new value.

        Args:
            val (float): The new value to update the running average.
        """
        self.total += val
        self.steps += 1

    def value(self):
        """Get the current value of the running average.

        Returns:
            float: The current running average value.
        """
        return self.total / float(self.steps)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k.

    Args:
        output (torch.Tensor): The model's output tensor.
        target (torch.Tensor): The target tensor.
        topk (tuple): A tuple of integers specifying the top-k values to compute.

    Returns:
        list: A list of accuracy values for each k in topk.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class KL_Loss(nn.Module):
    """
    Kullback-Leibler (KL) Divergence Loss with Temperature Scaling.

    This class represents the KL divergence loss with an optional temperature
    scaling parameter for softening the logits. It is commonly used in knowledge
    distillation between a student and a teacher model.

    Args:
        temperature (float, optional): The temperature parameter for softening
            the logits (default is 1).

    Attributes:
        T (float): The temperature parameter for temperature scaling.

    """

    def __init__(self, temperature=1):
        """
        Initialize the KL Divergence Loss.

        Args:
            temperature (float, optional): The temperature parameter for softening
                the logits (default is 1).

        """
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        """Compute the KL divergence loss between output_batch and teacher_outputs.

        Args:
            output_batch (torch.Tensor): The output tensor from the student model.
            teacher_outputs (torch.Tensor): The output tensor from the teacher model.

        Returns:
            torch.Tensor: The computed KL divergence loss.
        """
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = (
            self.T
            * self.T
            * nn.KLDivLoss(reduction="batchmean")(output_batch, teacher_outputs)
        )

        return loss



class CE_Loss(nn.Module):
    """
    Cross-Entropy Loss with Temperature Scaling.

    This class represents the cross-entropy loss with an optional temperature
    scaling parameter for softening the logits. It is commonly used in knowledge
    distillation between a student and a teacher model.

    Args:
        temperature (float, optional): The temperature parameter for softening
            the logits (default is 1).

    Attributes:
        T (float): The temperature parameter for temperature scaling.

    """

    def __init__(self, temperature=1):
        """
        Initialize the Cross-Entropy (CE) Loss.

        Args:
            temperature (float): The temperature parameter for softening the logits (default is 1).

        """
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        """Compute the cross-entropy loss between output_batch and teacher_outputs.

        Args:
            output_batch (torch.Tensor): The output tensor from the student model.
            teacher_outputs (torch.Tensor): The output tensor from the teacher model.

        Returns:
            torch.Tensor: The computed cross-entropy loss.
        """
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)

        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = (
            -self.T
            * self.T
            * torch.sum(torch.mul(output_batch, teacher_outputs))
            / teacher_outputs.size(0)
        )

        return loss

def save_dict_to_json(d, json_path):
    """Saves a dictionary of floats in a JSON file.

    Args:
        d (dict): A dictionary of float-castable values (np.float, int, float, etc.).
        json_path (str): Path to the JSON file where the dictionary will be saved.
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for JSON (it doesn't accept np.array, np.float, etc.)
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

# Filter out batch norm parameters and remove them from weight decay - gets us higher accuracy 93.2 -> 93.48
# https://arxiv.org/pdf/1807.11205.pdf
def bnwd_optim_params(model, model_params, master_params):
    """Split model parameters into two groups for optimization.

    This function separates model parameters into two groups: batch normalization parameters
    and remaining parameters. It sets the weight decay for batch normalization parameters to 0.

    Args:
        model (nn.Module): The neural network model.
        model_params (list): List of model parameters.
        master_params (list): List of master parameters.

    Returns:
        list: List of dictionaries specifying parameter groups for optimization.
    """
    bn_params, remaining_params = split_bn_params(model, model_params, master_params)
    return [{"params": bn_params, "weight_decay": 0}, {"params": remaining_params}]

def split_bn_params(model, model_params, master_params):
    """Split model parameters into batch normalization and remaining parameters.

    This function separates model parameters into two groups: batch normalization parameters
    and remaining parameters.

    Args:
        model (nn.Module): The neural network model.
        model_params (list): List of model parameters.
        master_params (list): List of master parameters.

    Returns:
        tuple: Two lists containing batch normalization parameters and remaining parameters.
    """
    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            return module.parameters()
        accum = set()
        for child in module.children():
            [accum.add(p) for p in get_bn_params(child)]
        return accum

    mod_bn_params = get_bn_params(model)
    zipped_params = list(zip(model_params, master_params))

    mas_bn_params = [
        p_mast for p_mod, p_mast in zipped_params if p_mod in mod_bn_params
    ]
    mas_rem_params = [
        p_mast for p_mod, p_mast in zipped_params if p_mod not in mod_bn_params
    ]
    return mas_bn_params, mas_rem_params
