import logging
from copy import deepcopy

import torch
import torch.nn as nn

""" model util """


def get_weights(state):
    """
    Returns list of weights from state_dict
    """
    if state is not None:
        return list(state.values())
    else:
        return None


def clear_optim_buffer(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            param_state = optimizer.state[p]
            # Reinitialize momentum buffer
            if "momentum_buffer" in param_state:
                param_state["momentum_buffer"].zero_()


""" cpu --- gpu """


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def move_to_cpu(model, optimizer):
    if str(next(model.parameters()).device) == "cpu":
        pass
    else:
        model = model.to("cpu")
        # optimizer_to(self.trainer.optimizer, 'cpu')
    if len(list(optimizer.state.values())) > 0:
        optimizer_to(optimizer, "cpu")
    return model


def move_to_gpu(model, optimizer, device):
    if str(next(model.parameters()).device) == "cpu":
        model = model.to(device)
    else:
        pass

    # logging.info(self.trainer.optimizer.state.values())
    if len(list(optimizer.state.values())) > 0:
        optimizer_to(optimizer, device)
    return model


""" get weights or grads """


def get_named_data(model, mode="MODEL", use_cuda=True):
    """
        getting the whole model and getting the gradients can be conducted
        by using different methods for reducing the communication.
        `model` choices: ['MODEL', 'GRAD', 'MODEL+GRAD'] 
    """
    if mode == "MODEL":
        own_state = model.cpu().state_dict()
        return own_state
    elif mode == "GRAD":
        grad_of_params = {}
        for name, parameter in model.named_parameters():
            # logging.info(f"Getting grads as named_grads: name:{name}, type(parameter): {type(parameter)},"+
            #             f" parameter.data.shape: {parameter.data.shape}, parameter.data.norm(): {parameter.data.norm()}"+
            #             f" parameter.grad.shape: {parameter.grad.shape}, parameter.grad.norm(): {parameter.grad.norm()}")
            if use_cuda:
                grad_of_params[name] = parameter.grad
            else:
                grad_of_params[name] = parameter.grad.cpu()
            # logging.info(f"Getting grads as named_grads: name:{name}, shape: {grad_of_params[name].shape}")
        return grad_of_params
    elif mode == "MODEL+GRAD":
        model_and_grad = {}
        for name, parameter in model.named_parameters():
            # if use_cuda:
            #     model_and_grad[name] = parameter.data
            #     model_and_grad[name+b'_gradient'] = parameter.grad
            # else:
            #     model_and_grad[name] = parameter.data.cpu()
            #     model_and_grad[name+b'_gradient'] = parameter.grad.cpu()
            if use_cuda:
                model_and_grad[name] = parameter.data
                model_and_grad[name + b"_gradient"] = parameter.grad
            else:
                model_and_grad[name] = parameter.data.cpu()
                model_and_grad[name + b"_gradient"] = parameter.grad.cpu()
        return model_and_grad


""" get bn params"""


def get_bn_params(prefix, module, use_cuda=True):
    bn_params = {}
    if use_cuda:
        bn_params[f"{prefix}.weight"] = module.weight
        bn_params[f"{prefix}.bias"] = module.bias
        bn_params[f"{prefix}.running_mean"] = module.running_mean
        bn_params[f"{prefix}.running_var"] = module.running_var
        bn_params[f"{prefix}.num_batches_tracked"] = module.num_batches_tracked
    else:
        bn_params[f"{prefix}.weight"] = module.weight.cpu()
        bn_params[f"{prefix}.bias"] = module.bias.cpu()
        bn_params[f"{prefix}.running_mean"] = module.running_mean.cpu()
        bn_params[f"{prefix}.running_var"] = module.running_var.cpu()
        bn_params[f"{prefix}.num_batches_tracked"] = module.num_batches_tracked
    return bn_params


def get_all_bn_params(model, use_cuda=True):
    all_bn_params = {}
    for module_name, module in model.named_modules():
        #     print(f"key:{key}, module, {module}")
        # logging.info(f"key:{key}, type(module) is nn.BatchNorm2d: {type(module) is nn.BatchNorm2d}")
        if type(module) is nn.BatchNorm2d:
            # logging.info(f"module.weight: {module.weight}")
            # logging.info(f"module.bias: {module.bias}")
            # logging.info(f"module.running_mean: {module.running_mean}")
            # logging.info(f"module.running_var: {module.running_var}")
            # logging.info(f"module.num_batches_tracked: {module.num_batches_tracked}")
            bn_params = get_bn_params(module_name, module, use_cuda=use_cuda)
            all_bn_params.update(bn_params)
    return all_bn_params


def check_bn_status(bn_module):
    logging.info(f"weight: {bn_module.weight[:10].mean()}")
    logging.info(f"bias: {bn_module.bias[:10].mean()}")
    logging.info(f"running_mean: {bn_module.running_mean[:10].mean()}")
    logging.info(f"running_var: {bn_module.running_var[:10].mean()}")
    logging.info(f"num_batches_tracked: {bn_module.num_batches_tracked}")
    logging.info(f"training: {bn_module.training}")


""" Average named params """


def average_named_params(named_params_list, average_weights_dict_list, inplace=True):
    """
        This is a weighted average operation.
        average_weights_dict_list: includes weights with respect to clients. Same for each param.
        inplace:  Whether change the first client's model inplace.
        Note: This function also can be used to average gradients.
    """
    # logging.info("################aggregate: %d" % len(named_params_list))

    if type(named_params_list[0]) is tuple or type(named_params_list[0]) is list:
        if inplace:
            (_, averaged_params) = named_params_list[0]
        else:
            (_, averaged_params) = deepcopy(named_params_list[0])
    else:
        if inplace:
            averaged_params = named_params_list[0]
        else:
            averaged_params = deepcopy(named_params_list[0])

    for k in averaged_params.keys():
        w_sum = 0.0
        for i in range(0, len(named_params_list)):
            if type(named_params_list[0]) is tuple or type(named_params_list[0]) is list:
                local_sample_number, local_named_params = named_params_list[i]
            else:
                local_named_params = named_params_list[i]
            # logging.debug("aggregating ---- local_sample_number/sum: {}/{}, ".format(
            #     local_sample_number, sum))
            w = average_weights_dict_list[i]
            w_sum += w
            # w = torch.full_like(local_named_params[k], w).detach()
            if "num_batches_tracked" in k:
                """Make it float first, then int."""
                # logging.info(f"local_named_params[{k}]: {local_named_params[k]} \
                #     w: {w}")
                if i == 0:
                    averaged_params[k] = local_named_params[k] * w
                else:
                    averaged_params[k] += (local_named_params[k] * w).to(averaged_params[k].device)
                    # logging.info(f"averaged_params[k]: {averaged_params[k].dtype} \
                    #     local_named_params[k]: {local_named_params[k].dtype}\
                    #     a:{a.dtype}")
                    # averaged_params[k] += local_named_params[k] * w
            else:
                if i == 0:
                    averaged_params[k] = (local_named_params[k] * w).type(averaged_params[k].dtype)
                else:
                    averaged_params[k] += (local_named_params[k].to(averaged_params[k].device) * w).type(
                        averaged_params[k].dtype
                    )

        if "num_batches_tracked" in k:
            """Make it float first, then int."""
            # logging.info(f"averaged_params[{k}]: {averaged_params[k]} \
            #     w_sum: {w_sum}")
            averaged_params[k] = averaged_params[k].type(local_named_params[k].dtype)

    return averaged_params


def get_average_weight(sample_num_list):
    # balance_sample_number_list = []
    average_weights_dict_list = []
    sum = 0

    for i in range(0, len(sample_num_list)):
        local_sample_number = sample_num_list[i]
        sum += local_sample_number

    for i in range(0, len(sample_num_list)):
        local_sample_number = sample_num_list[i]
        weight_by_sample_num = local_sample_number / sum
        average_weights_dict_list.append(weight_by_sample_num)

    return average_weights_dict_list


"""auxiliary."""


def check_device(data_src, device=None):
    if device is not None:
        if data_src.device is not device:
            return data_src.to(device)
        else:
            return data_src
    else:
        return data_src


"""Dif Utils"""


def get_diff_weights(weights1, weights2):
    """ Produce a direction from 'weights1' to 'weights2'."""
    if isinstance(weights1, list) and isinstance(weights2, list):
        return [w2 - w1 for (w1, w2) in zip(weights1, weights2)]
    elif isinstance(weights1, torch.Tensor) and isinstance(weights2, torch.Tensor):
        return weights2 - weights1
    else:
        raise NotImplementedError


def get_name_params_difference(named_parameters1, named_parameters2):
    """
        return named_parameters2 - named_parameters1
    """
    common_names = list(set(named_parameters1.keys()).intersection(set(named_parameters2.keys())))
    named_diff_parameters = {}
    for key in common_names:
        named_diff_parameters[key] = get_diff_weights(named_parameters1[key], named_parameters2[key])
    return named_diff_parameters
