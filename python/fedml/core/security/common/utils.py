import numpy as np
import torch
import torch.nn.functional as F


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


def compute_middle_point(alphas, model_list):
    """

    Args:
        alphas: weights of model_dict
        model_dict: a model submitted by a user

    Returns:

    """
    sum_batch = torch.zeros(model_list[0].shape)
    for a, a_batch_w in zip(alphas, model_list):
        sum_batch += a * a_batch_w
    return sum_batch


def get_total_sample_num(model_list):
    sample_num = 0
    for i in range(len(model_list)):
        local_sample_num, local_model_params = model_list[i]
        sample_num += local_sample_num
    return sample_num


def get_malicious_client_id_list(random_seed, client_num, malicious_client_num):
    if client_num == malicious_client_num:
        client_indexes = [client_index for client_index in range(client_num)]
    else:
        num_clients = min(malicious_client_num, client_num)
        np.random.seed(
            random_seed
        )  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(client_num), num_clients, replace=False)
    print("client_indexes = %s" % str(client_indexes))
    return client_indexes


def replace_original_class_with_target_class(
    data_labels, original_class_list=None, target_class_list=None
):
    """
    :param targets: Target class IDs
    :type targets: list
    :return: new class IDs
    """

    if (
        len(original_class_list) == 0
        or len(target_class_list) == 0
        or original_class_list is None
        or target_class_list is None
    ):
        return data_labels
    if len(original_class_list) != len(target_class_list):
        raise ValueError(
            "the length of the original class list is not equal to the length of the targeted class list"
        )
    if len(set(original_class_list)) < len(
        original_class_list
    ):  # no need to check the targeted classes
        raise ValueError("the original classes can not be same")

    for i in range(len(original_class_list)):
        for idx in range(len(data_labels)):
            if data_labels[idx] == original_class_list[i]:
                data_labels[idx] = target_class_list[i]
    return data_labels


def log_client_data_statistics(poisoned_client_ids, train_data_local_dict):
    """
    Logs all client data statistics.

    :param poisoned_client_ids: list of malicious clients
    :type poisoned_client_ids: list
    :param train_data_local_dict: distributed dataset
    :type train_data_local_dict: list(tuple)
    """
    for client_idx in range(len(train_data_local_dict)):
        if client_idx in poisoned_client_ids:
            targets_set = {}
            for _, (_, targets) in enumerate(train_data_local_dict[client_idx]):
                for target in targets.numpy():
                    if target not in targets_set.keys():
                        targets_set[target] = 1
                    else:
                        targets_set[target] += 1
            print("Client #{} has data distribution:".format(client_idx))
            for item in targets_set.items():
                print("target:{} num:{}".format(item[0], item[1]))


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))
