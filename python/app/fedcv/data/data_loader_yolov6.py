import logging
import torch
import numpy as np
from YOLOv6.yolov6.core.engine import Trainer

def partition_data(n_data, y_train, partition, n_nets):
    if partition == "homo":
        total_num = n_data
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    elif partition == "hetero":
        min_size = 0
        y_train = np.array(y_train)
        K = int(max(y_train) + 1)
        N = y_train.shape[0]
        net_dataidx_map = {}

        alpha = 0.5
        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

def load_partition_data_coco(args, yolo_args, cfg, device):
    client_number = args.client_num_in_total
    partition = args.partition_method

    meituan_trainer = Trainer(yolo_args, cfg, device)
    train_set = meituan_trainer.train_set
    train_dataloader_global = meituan_trainer.train_loader
    test_dataloader_global = meituan_trainer.val_loader
    n_data = len(train_set)
    nc = meituan_trainer.data_dict['nc']
    labels = train_set.labels
    y_train = [label[0][0] for label in labels]

    # load meituan_trainer
    net_dataidx_map = partition_data(
        n_data, y_train, partition=partition, n_nets=client_number
    )

    train_data_loader_dict = dict()
    test_data_loader_dict = dict()
    train_data_num_dict = dict()
    train_dataset_dict = dict()

    train_data_num = 0
    test_data_num = 0
    train_dataloader_global = None
    test_dataloader_global = None

    for client_idx in range(client_number):
        meituan_trainer = Trainer(yolo_args, cfg, device, net_dataidx_map[client_idx])

        train_dataset_dict[client_idx] = torch.utils.data.Subset(meituan_trainer.train_set, net_dataidx_map[client_idx])
        train_data_num_dict[client_idx] = len(train_dataset_dict[client_idx])
        train_data_loader_dict[client_idx] = None
        test_data_loader_dict[client_idx] = None
        train_data_num += train_data_num_dict[client_idx]

    return (
        train_data_num,
        test_data_num,
        train_dataloader_global,
        test_dataloader_global,
        train_data_num_dict,
        train_data_loader_dict,
        test_data_loader_dict,
        nc,
    ), net_dataidx_map
