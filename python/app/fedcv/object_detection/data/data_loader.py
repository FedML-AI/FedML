import logging

import copy
import os
import yaml
import math
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.datasets import create_dataloader
from pathlib import Path


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print(
            "WARNING: --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size


def partition_data(data_path, partition, n_nets):
    if os.path.isfile(data_path):
        with open(data_path) as f:
            data = f.readlines()
        n_data = len(data)
    else:
        n_data = len(os.listdir(data_path))
    if partition == "homo":
        total_num = n_data
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    elif partition == "hetero":
        _label_path = copy.deepcopy(data_path)
        label_path = _label_path.replace("images", "labels")
        net_dataidx_map = non_iid_coco(label_path, n_nets)
        # print(net_dataidx_map)

    return net_dataidx_map


def non_iid_coco(label_path, client_num):
    res_bin = {}
    label_path = Path(label_path)
    fs = os.listdir(label_path)
    fs = {i for i in fs if i.endswith(".txt")}
    bin_n = len(fs) // client_num
    # print(f"{len(fs)} files found, {bin_n} files per client")

    id2idx = {}  # coco128
    for i, f in enumerate(fs):
        id2idx[int(f.split(".")[0])] = i

    for b in range(bin_n - 1):
        res = {}
        for f in fs:
            if not f.endswith(".txt"):
                continue

            txt_path = os.path.join(label_path, f)
            txt_f = open(txt_path)
            for line in txt_f.readlines():
                line = line.strip("\n")
                l = line.split(" ")[0]
                if res.get(l) == None:
                    res[l] = set()
                else:
                    res[l].add(f)
            txt_f.close()

        sort_res = sorted(res.items(), key=lambda x: len(x[1]), reverse=True)
        # print(f"{b}th bin: {len(sort_res)} classes")
        # print(res)
        fs = fs - sort_res[0][1]
        # print(f"{len(fs)} files left")

        fs_id = [id2idx[int(i.split(".")[0])] for i in sort_res[0][1]]
        res_bin[b] = np.array(fs_id)

    fs_id = [int(i.split(".")[0]) for i in fs]
    res_bin[b + 1] = np.array(list(fs_id))
    return res_bin
    # print (res_bin)


def load_partition_data_coco(args, hyp, model):
    save_dir, epochs, batch_size, total_batch_size, weights = (
        Path(args.save_dir),
        args.epochs,
        args.batch_size,
        args.total_batch_size,
        args.weights,
    )

    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    train_path = data_dict["train"]
    test_path = data_dict["val"]
    train_path = os.path.expanduser(train_path)
    test_path = os.path.expanduser(test_path)

    nc, names = (
        (1, ["item"]) if args.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )  # number classes, names
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in args.img_size]

    client_number = args.client_num_in_total
    partition = args.partition_method

    # client_list = []

    net_dataidx_map = partition_data(
        train_path, partition=partition, n_nets=client_number
    )
    net_dataidx_map_test = partition_data(
        test_path, partition=partition, n_nets=client_number
    )
    train_data_loader_dict = dict()
    test_data_loader_dict = dict()
    train_data_num_dict = dict()
    train_dataset_dict = dict()

    # train_dataloader_global, train_dataset_global = create_dataloader(
    #     train_path,
    #     imgsz,
    #     batch_size,
    #     gs,
    #     args,
    #     hyp=hyp,
    #     rect=True,
    #     augment=True,
    #     workers=args.worker_num,
    # )
    # train_data_num = train_dataset_global.data_size

    # test_dataloader_global = create_dataloader(
    #     test_path,
    #     imgsz_test,
    #     total_batch_size,
    #     gs,
    #     args,  # testloader
    #     hyp=hyp,
    #     rect=True,
    #     pad=0.5,
    #     workers=args.worker_num,
    # )[0]

    # test_data_num = test_dataloader_global.dataset.data_size

    train_data_num = 0
    test_data_num = 0
    train_dataloader_global = None
    test_dataloader_global = None

    if args.process_id == 0:  # server
        pass
    else:
        client_idx = int(args.process_id) - 1

        logging.info(
            f"{client_idx}: net_dataidx_map trainer: {net_dataidx_map[client_idx]}"
        )
        dataloader, dataset = create_dataloader(
            train_path,
            imgsz,
            batch_size,
            gs,
            args,
            hyp=hyp,
            rect=True,
            augment=True,
            net_dataidx_map=net_dataidx_map[client_idx],
            workers=args.worker_num,
        )
        testloader = create_dataloader(
            test_path,
            imgsz_test,
            total_batch_size,
            gs,
            args,  # testloader
            hyp=hyp,
            rect=True,
            rank=-1,
            pad=0.5,
            net_dataidx_map=net_dataidx_map_test[client_idx],
            workers=args.worker_num,
        )[0]

        train_dataset_dict[client_idx] = dataset
        train_data_num_dict[client_idx] = len(dataset)
        train_data_loader_dict[client_idx] = dataloader
        test_data_loader_dict[client_idx] = testloader
        # client_list.append(
        #     Client(i, train_data_loader_dict[i], len(dataset), opt, device, model, tb_writer=tb_writer,
        #            hyp=hyp, wandb=wandb))
        #

    return (
        train_data_num,
        test_data_num,
        train_dataloader_global,
        test_dataloader_global,
        train_data_num_dict,
        train_data_loader_dict,
        test_data_loader_dict,
        nc,
    )
