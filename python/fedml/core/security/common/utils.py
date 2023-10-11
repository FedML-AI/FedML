import math
import random
import numpy as np
import torch
import torch.nn.functional as F


def vectorize_weight(state_dict):
    """
    Vectorizes the weight tensors in the given state_dict.

    Args:
        state_dict (OrderedDict): The state_dict containing model weights.

    Returns:
        torch.Tensor: A concatenated tensor of flattened weights.
    """
    weight_list = []
    for (k, v) in state_dict.items():
        if is_weight_param(k):
            weight_list.append(v.flatten())
    return torch.cat(weight_list)


def is_weight_param(k):
    """
    Checks if a parameter key is a weight parameter.

    Args:
        k (str): The parameter key.

    Returns:
        bool: True if the key corresponds to a weight parameter, False otherwise.
    """
    return (
        "running_mean" not in k
        and "running_var" not in k
        and "num_batches_tracked" not in k
    )


def compute_euclidean_distance(v1, v2, device='cpu'):
    """
    Computes the Euclidean distance between two tensors.

    Args:
        v1 (torch.Tensor): The first tensor.
        v2 (torch.Tensor): The second tensor.
        device (str): The device for computation (default is 'cpu').

    Returns:
        torch.Tensor: The Euclidean distance between the two tensors.
    """
    v1 = v1.to(device)
    v2 = v2.to(device)
    return (v1 - v2).norm()


def compute_model_norm(model):
    """
    Computes the norm of a model's weights.

    Args:
        model: The model.

    Returns:
        torch.Tensor: The norm of the model's weights.
    """
    return vectorize_weight(model).norm()


def compute_middle_point(alphas, model_list):
    """
    Computes the weighted sum of model weights.

    Args:
        alphas (list): List of weights.
        model_list (list): List of model weights.

    Returns:
        numpy.ndarray: The weighted sum of model weights.
    """
    sum_batch = torch.zeros(model_list[0].shape)
    for a, a_batch_w in zip(alphas, model_list):
        sum_batch += a * a_batch_w.float().cpu().numpy()
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
    """
    Calculates the total number of samples across multiple clients.

    Args:
        model_list (list): List of tuples containing local sample numbers and model parameters.

    Returns:
        int: Total number of samples.
    """
    sample_num = 0
    for i in range(len(model_list)):
        local_sample_num, local_model_params = model_list[i]
        sample_num += local_sample_num
    return sample_num


def get_malicious_client_id_list(random_seed, client_num, malicious_client_num):
    """
    Generates a list of malicious client IDs.

    Args:
        random_seed (int): Random seed for reproducibility.
        client_num (int): Total number of clients.
        malicious_client_num (int): Number of malicious clients to generate.

    Returns:
        list: List of malicious client IDs.
    """
    if client_num == malicious_client_num:
        client_indexes = [client_index for client_index in range(client_num)]
    else:
        num_clients = min(malicious_client_num, client_num)
        np.random.seed(
            random_seed
        )  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(
            range(client_num), num_clients, replace=False)
    print("malicious client_indexes = %s" % str(client_indexes))
    return client_indexes


def replace_original_class_with_target_class(
        data_labels, original_class_list=None, target_class_list=None
):
    """
    Replaces original class labels in data_labels with corresponding target class labels.

    Args:
        data_labels (list): List of class labels.
        original_class_list (list): List of original class labels to be replaced.
        target_class_list (list): List of target class labels to replace with.

    Returns:
        list: Updated list of class labels.
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
    Logs data distribution statistics for each client in the dataset.

    Args:
        poisoned_client_ids (list): List of malicious client IDs.
        train_data_local_dict (list): Distributed dataset.
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


def get_client_data_stat(local_dataset):
    """
    Prints data distribution statistics for a local dataset.

    Args:
        local_dataset (Iterable): Local dataset.

    """
    print("-==========================")
    targets_set = {}
    for batch_idx, (data, targets) in enumerate(local_dataset):
        for t in targets.tolist():
            if t in targets_set.keys():
                targets_set[t] += 1
            else:
                targets_set[t] = 1
            # if t not in targets_set.keys():
            #     targets_set[t] = 1
            # else:
            #     targets_set[t] += 1
    total_counter = 0
    # for item in targets_set.items():
    #     print("------target:{} num:{}".format(item[0], item[1]))
    #     total_counter += item[1]
    # print(f"total counter = {total_counter}")
    #
    # targets_set = {}
    # for batch_idx, (data, targets) in enumerate(local_dataset):
    #     for t in targets.tolist():
    #         if t in targets_set.keys():
    #             targets_set[t] += 1
    #         else:
    #             targets_set[t] = 1
    #         # if t not in targets_set.keys():
    #         #     targets_set[t] = 1
    #         # else:
    #         #     targets_set[t] += 1
    # total_counter = 0
    for item in targets_set.items():
        print("------target:{} num:{}".format(item[0], item[1]))
        total_counter += item[1]
    print(f"total counter = {total_counter}")


def cross_entropy_for_onehot(pred, target):
    """
    Computes the cross-entropy loss between predicted and target one-hot encoded vectors.

    Args:
        pred (torch.Tensor): Predicted logit values.
        target (torch.Tensor): Target one-hot encoded vectors.

    Returns:
        torch.Tensor: Cross-entropy loss.

    """
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def label_to_onehot(target, num_classes=100):
    """
    Converts class labels to one-hot encoded vectors.

    Args:
        target (torch.Tensor): Class labels.
        num_classes (int, optional): Number of classes. Defaults to 100.

    Returns:
        torch.Tensor: One-hot encoded vectors.

    """
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(
        0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def trimmed_mean(model_list, trimmed_num):
    """
    Trims the list of models by removing a specified number of models from both ends.

    Args:
        model_list (list): List of model tuples containing local sample numbers and gradients.
        trimmed_num (int): Number of models to trim from each end.

    Returns:
        list: Trimmed list of models.

    """
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
    # sort by coordinate-wise scores
    temp_model_list.sort(key=lambda grad: grad[2])
    temp_model_list = temp_model_list[trimmed_num: len(
        model_list) - trimmed_num]
    model_list = [(t[0], t[1]) for t in temp_model_list]
    return model_list


def compute_a_score(local_sample_number):
    """
    Compute a score for a client based on its local sample number.

    Args:
        local_sample_number (int): Number of local samples for a client.

    Returns:
        int: A score for the client.

    """
    # todo: change to coordinate-wise score
    return local_sample_number


def compute_krum_score(vec_grad_list, client_num_after_trim, p=2):
    """
    Compute Krum scores for clients based on their gradients.

    Args:
        vec_grad_list (list): List of gradient vectors for each client.
        client_num_after_trim (int): Number of clients to consider.
        p (int, optional): Power parameter for distance calculation. Defaults to 2.

    Returns:
        list: List of Krum scores for each client.

    """
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
    """
    Compute the mean (mu) and standard deviation (sigma) of a list of scores.

    Args:
        score_list (list): List of scores.

    Returns:
        Tuple[float, float]: Mean (mu) and standard deviation (sigma).

    """
    n = len(score_list)
    mu = sum(list(score_list)) / n
    temp = 0

    for i in range(len(score_list)):
        temp = (((score_list[i] - mu) ** 2) / (n - 1)) + temp
    sigma = math.sqrt(temp)
    return mu, sigma


def sample_some_clients(client_num, sampled_client_num):
    """
    Sample a specified number of clients from the total number of clients.

    Args:
        client_num (int): Total number of clients.
        sampled_client_num (int): Number of clients to sample.

    Returns:
        list: List of sampled client indices.

    """
    return random.sample(range(client_num), sampled_client_num)
