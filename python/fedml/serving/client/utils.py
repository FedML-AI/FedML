from collections import OrderedDict


# ref: https://discuss.pytorch.org/t/failed-to-load-model-trained-by-ddp-for-inference/84841/2?u=amir_zsh
def convert_model_params_from_ddp(ddp_model_params):
    """
    Convert model parameters from DistributedDataParallel (DDP) format to a regular format.

    Args:
        ddp_model_params (OrderedDict): Model parameters in DDP format.

    Returns:
        OrderedDict: Model parameters in regular format.

    Example:
        >>> ddp_params = OrderedDict([('module.conv1.weight', tensor), ('module.fc1.weight', tensor)])
        >>> regular_params = convert_model_params_from_ddp(ddp_params)
    """
    model_params = OrderedDict()
    for k, v in ddp_model_params.items():
        name = k[7:]  # Remove 'module.' of DataParallel/DistributedDataParallel
        model_params[name] = v
    return model_params


def convert_model_params_to_ddp(model_params):
    """
    Convert model parameters from a regular format to DistributedDataParallel (DDP) format.

    Args:
        model_params (OrderedDict): Model parameters in regular format.

    Returns:
        OrderedDict: Model parameters in DDP format.

    Example:
        >>> regular_params = OrderedDict([('conv1.weight', tensor), ('fc1.weight', tensor)])
        >>> ddp_params = convert_model_params_to_ddp(regular_params)
    """
    ddp_model_params = OrderedDict()
    for k, v in model_params.items():
        name = f"module.{k}"  # Add 'module.' for DataParallel/DistributedDataParallel
        ddp_model_params[name] = v
    return ddp_model_params
