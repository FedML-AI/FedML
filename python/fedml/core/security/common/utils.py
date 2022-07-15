import numpy as np
import torch
import pickle


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


def replace_original_class_with_target_class(data_labels, original_class, target_class):
    """
    :param targets: Target class IDs
    :type targets: list
    :return: new class IDs
    """
    if original_class == target_class or original_class is None or target_class is None:
        return data_labels
    for idx in range(len(data_labels)):
        if data_labels[idx] == original_class:
            data_labels[idx] = target_class
    return data_labels


def load_data_loader_from_file(filename):
    """
    Loads DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param filename: string
    """
    print("Loading data loader from file: {}".format(filename))

    with open(filename, "rb") as f:
        return load_saved_data_loader(f)


def load_saved_data_loader(file_obj):
    return pickle.load(file_obj)
