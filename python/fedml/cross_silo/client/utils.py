from collections import OrderedDict


# ref: https://discuss.pytorch.org/t/failed-to-load-model-trained-by-ddp-for-inference/84841/2?u=amir_zsh
def convert_model_params_from_ddp(ddp_model_params):
    """
    Convert model parameters from DataParallel/DistributedDataParallel format to a regular model format.

    Args:
        ddp_model_params (dict): Model parameters in DataParallel/DistributedDataParallel format.

    Returns:
        OrderedDict: Model parameters in the regular format.
    """
    model_params = OrderedDict()
    for k, v in ddp_model_params.items():
        name = k[7:]  # Remove 'module.' of DataParallel/DistributedDataParallel
        model_params[name] = v
    return model_params


def convert_model_params_to_ddp(model_params):
    """
    Convert model parameters from a regular format to DataParallel/DistributedDataParallel format.

    Args:
        model_params (dict): Model parameters in the regular format.

    Returns:
        OrderedDict: Model parameters in DataParallel/DistributedDataParallel format.
    """
    ddp_model_params = OrderedDict()
    for k, v in model_params.items():
        # Add 'module.' for DataParallel/DistributedDataParallel
        name = f"module.{k}"
        ddp_model_params[name] = v
    return ddp_model_params


def check_method_override(cls_obj, method_name: str) -> bool:
    """
    Check if a method has been overridden by a class.

    Args:
        cls_obj (object): The class object.
        method_name (str): The name of the method to check for override.

    Returns:
        bool: True if the method has been overridden, False otherwise.
    """
    # Check if method has been overridden by class
    return (
        method_name in cls_obj.__class__.__dict__ and
        hasattr(cls_obj, method_name) and
        callable(getattr(cls_obj, method_name))
    )
