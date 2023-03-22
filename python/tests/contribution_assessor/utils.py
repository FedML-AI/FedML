import collections
import copy
import pickle

import torch


def create_fake_global_w_local_w_MNIST():
    local_w = dict()
    local_w["linear.weight"] = torch.FloatTensor(
        [
            [-0.0003, 0.0192, -0.0294, 0.0219, 0.0037, 0.0021],
            [-0.0198, -0.0150, -0.0104, -0.0203, -0.0060, -0.0299],
            [-0.0201, 0.0149, -0.0333, -0.0203, 0.0012, 0.0080],
            [0.0237, 0.0103, -0.0219, 0.0088, -0.0009, 0.0009],
            [0.0144, -0.0336, -0.0346, -0.0222, -0.0025, -0.0138],
            [-0.0196, -0.0118, 0.0230, -0.0202, 0.0172, 0.0355],
        ]
    )
    local_w["linear.bias"] = torch.FloatTensor([-0.0753, -0.0589, -0.0907, -0.0672, 0.1159, 0.0290])
    global_w = dict()
    global_w["linear.weight"] = torch.FloatTensor(
        [
            [-0.0003, 0.0192, -0.0294, 0.0219, 0.0037, 0.0021],
            [-0.0198, -0.0150, -0.0104, -0.0203, -0.0060, -0.0299],
            [-0.0201, 0.0149, -0.0333, -0.0203, 0.0012, 0.0080],
            [0.0237, 0.0103, -0.0219, 0.0088, -0.0009, 0.0009],
            [0.0144, -0.0336, -0.0346, -0.0222, -0.0025, -0.0138],
            [-0.0196, -0.0118, 0.0230, -0.0202, 0.0172, 0.0355],
        ]
    )
    global_w["linear.bias"] = torch.FloatTensor([-0.0745, -0.0578, -0.0899, -0.0662, 0.1122, 0.0295])
    return local_w, global_w


def create_fake_vectors():
    return torch.FloatTensor([[0, 0, 0], [0, 0, 0], [1, 1, 1]]), torch.FloatTensor([[3, 1, 1], [5, 5, 5], [2, 2, 2]])


def create_fake_local_w_global_w():
    local_w = dict()
    local_w["linear.weight"] = torch.FloatTensor([[0, 0, 0], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1]])
    local_w["linear.bias"] = torch.FloatTensor([0, 0, 0.1])
    global_w = collections.OrderedDict()
    global_w["linear.weight"] = torch.FloatTensor([[0.3, 0.1, 0.1], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]])
    global_w["linear.bias"] = torch.FloatTensor([0.1, 0.1, 0.1])

    return local_w, global_w


def create_fake_model_list(active_worker_num):
    a_local_w = dict()
    a_local_w["linear.weight"] = torch.FloatTensor(
        [[0.1, 0.2, 0.2, 0.1], [0.15, 0.12, 0.02, 0.2], [0.3, 0.01, 0.21, 0.11]]
    )
    a_local_w["linear.bias"] = torch.FloatTensor([0.01, 0.19, 0.21])
    model_list = []
    for i in range(active_worker_num):
        local_w = dict()
        local_w["linear.weight"] = (i + 1) * a_local_w["linear.weight"]
        local_w["linear.bias"] = (i + 1) * a_local_w["linear.bias"]
        model_list.append((i + 10, local_w))  # add a random sample num
    return model_list


def create_fake_model_list(active_worker_num):
    a_local_w = dict()
    a_local_w["linear.weight"] = torch.FloatTensor(
        [[0.1, 0.2, 0.2, 0.1], [0.15, 0.12, 0.02, 0.2], [0.3, 0.01, 0.21, 0.11]]
    )
    a_local_w["linear.bias"] = torch.FloatTensor([0.01, 0.19, 0.21])
    model_list = []
    for i in range(active_worker_num):
        local_w = dict()
        local_w["linear.weight"] = (i+1) * a_local_w["linear.weight"]
        local_w["linear.bias"] = (i+1) * a_local_w["linear.bias"]
        model_list.append((i + 10, local_w))  # add a random sample num
    return model_list


def create_fake_model_list_MNIST(active_worker_num):
    a_local_w = dict()
    a_local_w["linear.weight"] = torch.FloatTensor(
        [
            [-0.0003, 0.0192, -0.0294, 0.0219, 0.0037, 0.0021],
            [-0.0198, -0.0150, -0.0104, -0.0203, -0.0060, -0.0299],
            [-0.0201, 0.0149, -0.0333, -0.0203, 0.0012, 0.0080],
            [0.0237, 0.0103, -0.0219, 0.0088, -0.0009, 0.0009],
            [0.0144, -0.0336, -0.0346, -0.0222, -0.0025, -0.0138],
            [-0.0196, -0.0118, 0.0230, -0.0202, 0.0172, 0.0355],
        ]
    )
    a_local_w["linear.bias"] = torch.FloatTensor([-0.0753, -0.0589, -0.0907, -0.0672, 0.1159, 0.0290])

    b_local_w = dict()
    b_local_w["linear.weight"] = torch.FloatTensor(
        [
            [-0.0203, 0.0192, -0.0294, 0.0219, 0.0037, 0.0021],
            [-0.0198, -0.1150, -0.1104, -0.0203, -0.0060, -0.0299],
            [-0.0201, 0.0149, -0.0333, -0.0203, 0.0012, 0.0080],
            [0.0237, 0.0103, -0.0219, 0.0088, -0.1009, 0.0109],
            [0.0144, -0.0136, -0.0146, -0.0222, -0.0025, -0.0138],
            [-0.0196, -0.0118, 0.0230, -0.0202, 0.2172, 0.0355],
        ]
    )
    b_local_w["linear.bias"] = torch.FloatTensor([-0.1153, -0.0589, -0.1907, -0.0672, 0.1159, 0.0290])

    c_local_w = dict()
    c_local_w["linear.weight"] = torch.FloatTensor(
        [
            [-0.0003, 0.0192, -0.0294, 0.0219, 0.0037, 0.0021],
            [-0.0198, -0.1150, -0.0104, -0.0203, -0.0060, -0.0299],
            [-0.0201, 0.0149, -0.3333, -0.0203, 0.0012, 0.1180],
            [0.0237, 0.0103, -0.2219, 0.0088, -0.0009, 0.1009],
            [0.0144, -0.0336, -0.1346, -0.0222, -0.0025, -0.1138],
            [-0.0196, -0.0118, 0.0230, -0.2202, 0.0172, 0.1355],
        ]
    )
    c_local_w["linear.bias"] = torch.FloatTensor([-0.0753, -0.0189, -0.0207, -0.2672, 0.1159, 0.0290])
    model_list = []
    for i in range(active_worker_num):
        if i % 3 == 0:
            model_list.append((20 + i, copy.deepcopy(a_local_w)))  # add a random sample num
        if i % 3 == 1:
            model_list.append((30 + i, copy.deepcopy(b_local_w)))  # add a random sample num
        if i % 3 == 2:
            model_list.append((25 + i, copy.deepcopy(c_local_w)))  # add a random sample num
    return model_list


def create_fake_gradient_Cifar100(file_path="fake_data/fake_gradient_Cifar100.pkl"):
    gradient = pickle.load(open(file_path, "rb"))
    return gradient


def create_fake_model_Cifar100(file_path="fake_data/fake_model_Cifar100.pt"):
    model = torch.load(file_path)
    return model


def create_fake_data_Cifar100(file_path="fake_data/fake_data_Cifar100.pkl"):
    data = torch.load(file_path)
    return data


def create_fake_dataloader():
    x = torch.rand(10, 3)
    y = torch.linspace(10, 1, 10)
    return (x, y)
