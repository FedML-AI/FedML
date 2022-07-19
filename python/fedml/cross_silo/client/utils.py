from collections import OrderedDict


# ref: https://discuss.pytorch.org/t/failed-to-load-model-trained-by-ddp-for-inference/84841/2?u=amir_zsh
def convert_model_params_from_ddp(ddp_model_params):
    model_params = OrderedDict()
    for k, v in ddp_model_params.items():
        name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        model_params[name] = v
    return model_params


def convert_model_params_to_ddp(ddp_model_params):
    model_params = OrderedDict()
    for k, v in ddp_model_params.items():
        name = f"module.{k}"  # add 'module.' of DataParallel/DistributedDataParallel
        model_params[name] = v
    return model_params
