import logging
import torch.utils.data as data
from fedml.data.cifar10.datasets import CIFAR10_truncated
import numpy as np
from  fedml.data.cifar10.data_loader import partition_data, _data_transforms_cifar10
"""
Changing fedml.data.cifar10.data_loader parts so that it is compatible with distributed trainers
"""

def load_data(args):
    (
    train_data_num,
    test_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    class_num,
    ) = load_partition_data_cifar10(
        args.dataset,
        args.data_cache_dir,
        args.partition_method,
        args.partition_alpha,
        args.worker_num,
        args.batch_size,
        args.silo_proc_num,
    )
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num




# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, n_dist_worker=1):
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs, n_dist_worker)




def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None, n_dist_worker=1):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(
        datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True
    )
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_data_loaders = []

    for rank in range(n_dist_worker):
            train_sampler = data.distributed.DistributedSampler(
                dataset=train_ds, 
                num_replicas=n_dist_worker, 
                rank=rank,
                shuffle=True,
                drop_last=False
            )
            train_dl = data.DataLoader(
                dataset=train_ds, batch_size=train_bs, shuffle=(train_sampler is None), drop_last=False,
                sampler=train_sampler,
            )
            test_dl = data.DataLoader(
                dataset=test_ds,batch_size=test_bs, shuffle=False, drop_last=False
            )        
            train_data_loaders.append(train_dl)


    return train_data_loaders, test_dl



def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, n_dist_trainer=0):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs, n_dist_trainer)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num














