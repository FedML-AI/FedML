import math
import random
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


def compute_model_norm(model):
    return vectorize_weight(model).norm()


def compute_middle_point(alphas, model_list):
    """
    alphas: weights of model_dict
    model_dict: a model submitted by a user
    """
    sum_batch = torch.zeros(model_list[0].shape)
    for a, a_batch_w in zip(alphas, model_list):
        sum_batch += a * a_batch_w
    return sum_batch


def compute_geometric_median(weights, client_grads):
    """
    Implementation of Weiszfeld's algorithm.
    Reference:  (1) https://github.com/krishnap25/RFA/blob/master/models/model.py
                (2) https://github.com/bladesteam/blades/blob/master/src/blades/aggregators/geomed.py
    our contribution: (07/01/2022)
    1) fix one bug in (1): (1) can not correctly compute a weighted average. The function weighted_average_oracle
    returns zero.
    2) fix one bug in (2): (2) can not correctly handle multidimensional tensors.
    3) reconstruct the code.
    """
    eps = 1e-5
    ftol = 1e-10
    middle_point = compute_middle_point(weights, client_grads)
    val = sum(
        [
            alpha * compute_euclidean_distance(middle_point, p)
            for alpha, p in zip(weights, client_grads)
        ]
    )
    for i in range(100):
        prev_median, prev_obj_val = middle_point, val
        weights = np.asarray(
            [
                max(
                    eps,
                    alpha
                    / max(eps, compute_euclidean_distance(middle_point, a_batch_w)),
                )
                for alpha, a_batch_w in zip(weights, client_grads)
            ]
        )
        weights = weights / weights.sum()
        middle_point = compute_middle_point(weights, client_grads)
        val = sum(
            [
                alpha * compute_euclidean_distance(middle_point, p)
                for alpha, p in zip(weights, client_grads)
            ]
        )
        if abs(prev_obj_val - val) < ftol * val:
            break
    return middle_point


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


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def trimmed_mean(model_list, trimmed_num):
    temp_model_list = []
    for i in range(0, len(model_list)):
        local_sample_num, client_grad = model_list[i]
        temp_model_list.append(
            (
                local_sample_num,
                client_grad,
                compute_a_score(local_sample_num),
            )
        )
    temp_model_list.sort(key=lambda grad: grad[2])  # sort by coordinate-wise scores
    temp_model_list = temp_model_list[trimmed_num: len(model_list) - trimmed_num]
    model_list = [(t[0], t[1]) for t in temp_model_list]
    return model_list


def compute_a_score(local_sample_number):
    # todo: change to coordinate-wise score
    return local_sample_number


def compute_krum_score(vec_grad_list, client_num_after_trim, p=2):
    krum_scores = []
    num_client = len(vec_grad_list)
    for i in range(0, num_client):
        dists = []
        for j in range(0, num_client):
            if i != j:
                dists.append(
                    compute_euclidean_distance(
                        torch.Tensor(vec_grad_list[i]),
                        torch.Tensor(vec_grad_list[j]),
                    ).item() ** p
                )
        dists.sort()  # ascending
        score = dists[0:client_num_after_trim]
        krum_scores.append(sum(score))
    return krum_scores


def compute_gaussian_distribution(score_list):
    n = len(score_list)
    mu = sum(list(score_list)) / n
    temp = 0

    for i in range(len(score_list)):
        temp = (((score_list[i] - mu) ** 2) / (n - 1)) + temp
    sigma = math.sqrt(temp)
    return mu, sigma


def sample_some_clients(client_num, sampled_client_num):
    return random.sample(range(client_num), sampled_client_num)
