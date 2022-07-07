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


def compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def compute_middle_point(alphas, model_dict):
    """

    Args:
        alphas: weights of model_dict
        model_dict: a model submitted by a user

    Returns:

    """
    sum_batch = torch.zeros(model_dict[0].shape)
    for a, a_batch_w in zip(alphas, model_dict):
        sum_batch += a * a_batch_w
    return sum_batch


def get_total_sample_num(model_list):
    sample_num = 0
    for i in range(len(model_list)):
        local_sample_num, local_model_params = model_list[i]
        sample_num += local_sample_num
    return sample_num