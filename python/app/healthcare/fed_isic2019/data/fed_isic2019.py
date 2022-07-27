import logging

import flamby
import torch
from flamby.datasets.fed_isic2019.dataset import FedIsic2019


def load_partition_fed_isic2019(args):
    train_data_num = 0
    test_data_num = 0
    train_data_global = None
    test_data_global = None
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    nc = 0

    if args.process_id == 0:  # server
        pass
    else:  # client
        logging.info(f"load center {int(args.process_id)-1} data")
        client_idx = int(args.process_id) - 1
        train_dataset = FedIsic2019(
            center=client_idx,
            train=True,
            debug=args.debug,
            data_path=args.data_cache_dir,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.worker_num,
        )
        test_dataset = FedIsic2019(
            center=client_idx,
            train=False,
            debug=args.debug,
            data_path=args.data_cache_dir,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.worker_num,
            drop_last=True,
        )
        data_local_num_dict[client_idx] = len(train_dataset)
        train_data_local_dict[client_idx] = train_dataloader
        test_data_local_dict[client_idx] = test_dataloader

    # logging.info(f"train_data_num: {train_data_num}")
    # logging.info(f"test_data_num: {test_data_num}")
    # logging.info(f"train_data_global: {train_data_global}")
    # logging.info(f"test_data_global: {test_data_global}")
    # logging.info(f"data_local_num_dict: {data_local_num_dict}")
    # logging.info(f"train_data_local_dict: {train_data_local_dict}")
    # logging.info(f"test_data_local_dict: {test_data_local_dict}")
    # logging.info(f"nc: {nc}")

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        nc,
    )
