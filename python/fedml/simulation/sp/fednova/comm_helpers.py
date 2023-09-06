import torch

"""
FedNova Optimizer comm_helper implementation cited from https://github.com/JYWa/FedNova/tree/master
"""


def flatten_tensors(tensors):
    """
    Flatten a list of dense tensors into a contiguous 1D buffer.

    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    This function takes a list of dense tensors and flattens them into a single
    contiguous 1D buffer. It assumes that all input tensors are of the same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): The list of dense tensors to flatten.

    Returns:
        Tensor: A 1D buffer containing the flattened input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Unflatten a flat buffer into a list of tensors using their original sizes.

    This function takes a flat buffer and unflattens it into a list of tensors using
    the sizes of the original tensors. It assumes that all input tensors are of the
    same dense type and that the flat buffer was generated using `flatten_tensors`.

    Args:
        flat (Tensor): The flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): The dense tensors whose sizes will be used to
            unflatten the flat buffer.

    Returns:
        tuple: Unflattened dense tensors with sizes same as `tensors` and values from `flat`.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def communicate(tensors, communication_op):
    """
    Communicate a list of tensors using a specified communication operation.

    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    This function takes a list of tensors and communicates them using a specified
    communication operation. It assumes that the communication_op can handle the
    provided tensors appropriately, such as performing an all-reduce operation.

    Args:
        tensors (Iterable[Tensor]): List of tensors to be communicated.
        communication_op: A method or partial object which takes a tensor as input
            and communicates it. It can be a partial object around something like
            `torch.distributed.all_reduce`.
    """
    flat_tensor = flatten_tensors(tensors)
    communication_op(tensor=flat_tensor)
    for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
        t.set_(f)
