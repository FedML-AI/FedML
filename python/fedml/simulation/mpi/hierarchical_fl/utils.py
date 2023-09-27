import os
import time

import numpy as np
import torch
import wandb

from sklearn.cluster import KMeans


def cal_mixing_consensus_speed(topo_weight_matrix, global_round_idx):
    n_rows, n_cols = np.shape(topo_weight_matrix)
    assert n_rows == n_cols
    A = np.array(topo_weight_matrix) - 1 / n_rows
    p = 1 - np.linalg.norm(A, ord=2) ** 2
    wandb.log({"Groups/p": p, "comm_round": global_round_idx})
    return p


def visualize_group_detail(group_to_client_indexes, train_data_local_dict, train_data_local_num_dict, class_num):

    xs = [i for i in range(class_num)]
    ys = []
    keys = []
    for group_idx in range(len(group_to_client_indexes)):
        data_size = 0
        group_y_train = []
        for client_id in group_to_client_indexes[group_idx]:
            data_size += train_data_local_num_dict[client_id]
            y_train = torch.concat([y for _, y in train_data_local_dict[client_id]]).tolist()
            group_y_train.extend(y_train)

        labels, counts = np.unique(group_y_train, return_counts=True)

        count_vector = np.zeros(class_num)
        count_vector[labels] = counts
        ys.append(count_vector/count_vector.sum())
        keys.append("Group {}".format(group_idx))

        wandb.log({"Groups/Client_num": len(group_to_client_indexes[group_idx]), "group_id": group_idx})
        wandb.log({"Groups/Data_size": data_size, "group_id": group_idx})

    wandb.log({"Groups/Data_distribution":
                   wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title="Data distribution", xname="Label")}
              )


def hetero_partition_groups(clients_type_list, num_groups, alpha=0.5):
    min_size = 0
    num_type = np.unique(clients_type_list).size
    N = len(clients_type_list)
    group_to_client_indexes = {}
    while min_size < 10:
        idx_batch = [[] for _ in range(num_groups)]
        # for each type in clients
        for k in range(num_type):
            idx_k = np.where(np.array(clients_type_list) == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_groups))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_groups) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    group_indexes = [0 for _ in range(N)]
    for j in range(num_groups):
        np.random.shuffle(idx_batch[j])
        group_to_client_indexes[j] = idx_batch[j]
        for client_id in group_to_client_indexes[j]:
            group_indexes[client_id] = j

    return group_indexes, group_to_client_indexes


def analyze_clients_type(train_data_local_dict, class_num, num_type=5):
    client_feature_list = []
    for i in range(len(train_data_local_dict)):
        y_train = torch.concat([y for _, y in train_data_local_dict[i]])
        labels, counts = torch.unique(y_train, return_counts=True)
        data_feature = np.zeros(class_num)
        total = 0
        for label, count in zip(labels, counts):
            data_feature[label.item()] = count.item()
            total += count.item()
        data_feature /= total
        client_feature_list.append(data_feature)

    kmeans = KMeans(n_clusters=num_type, random_state=0, n_init="auto").fit(client_feature_list)



    # for k in range(num_type):
    #     tmp = []
    #     for i, j in enumerate(kmeans.labels_):
    #         if j == k:
    #             indexes = np.where(np.array(client_feature_list[i]) > 0)
    #             tmp.extend(indexes[0].tolist())
    #     print(np.unique(tmp))
    #
    # exit(0)
    return kmeans.labels_


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    pipe_path = "./tmp/fedml"
    os.system("mkdir -p ./tmp/; touch ./tmp/fedml")
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, "w") as pipe:
        pipe.write("training is finished! \n%s\n" % (str(args)))
    time.sleep(3)
