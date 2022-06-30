import torch


def vectorize_weight(state_dict):
    weight_list = []
    for (k, v) in state_dict.items():
        if is_weight_param(k):
            weight_list.append(v.flatten())
    return torch.cat(weight_list)


def is_weight_param(k):
    return (
        "running_mean" not in k
        and "running_var" not in k
        and "num_batches_tracked" not in k
    )
