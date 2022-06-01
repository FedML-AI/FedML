import logging


import os
import yaml
import math
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from .datasets import create_dataloader
from pathlib import Path


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
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
    elif partition == 'hetero':
        _label_path=copy.deepcopy(train_path)
        label_path=_label_path.replace('images','label')
        net_data_idx_map = non_iid_coco(label_path, n_nets)


    return net_dataidx_map

def non_iid_coco(label_path, client_num):
    res_bin = {}
    fs = os.listdir(label_path)
    fs = {i for i in fs if i.endswith('.txt')}

    for b in range(bin_n-1):
        res = {}
        for f in fs:
            if not f.endswith('.txt'):
                continue

            txt_f= open(os.path.join(txt_path, f))
            for line in txt_f.readlines():
                line = line.strip('\n')
                l = line.split(' ')[0]
                if res.get(l) == None:
                    res[l] = set()
                else:
                    res[l].add(f)
            txt_f.close()

        sort_res = sorted(res.items(), key=lambda x:len(x[1]), reverse=True)
        fs = fs - sort_res[0][1]

        fs_id = [int(i.split('.')[0]) for i in sort_res[0][1]]
        res_bin[b] = np.array(fs_id)

    fs_id = [int(i.split('.')[0]) for i in fs]
    res_bin[b+1] = np.array(list(fs_id))
    return res_bin
    #print (res_bin)
    
    
def load_partition_data_coco(opt, hyp, model):
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]



    client_number = opt.client_num_in_total
    partition = opt.partition_method

    # client_list = []

    net_dataidx_map = partition_data(train_path, partition=partition, n_nets=client_number)
    net_dataidx_map_test = partition_data(test_path, partition=partition, n_nets=client_number)
    train_data_loader_dict = dict()
    test_data_loader_dict = dict()
    train_data_num_dict = dict()
    train_dataset_dict = dict()

    train_dataloader_global, train_dataset_global = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights)
    train_data_num = len(train_dataset_global)

    test_dataloader_global = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,  # testloader
                                   hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                   rank=-1, world_size=opt.world_size, workers=opt.workers, pad=0.5)[0]

    test_data_num = len(test_dataloader_global.dataset)


    for i in range(client_number):
        print("net_dataidx_map trainer:", net_dataidx_map[i])
        dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                                rank=rank,
                                                world_size=opt.world_size, workers=opt.workers,
                                                image_weights=opt.image_weights,
                                                net_dataidx_map=net_dataidx_map[i])
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers, pad=0.5, net_dataidx_map=net_dataidx_map_test[i])[0]



        train_dataset_dict[i] = dataset
        train_data_num_dict[i] = len(dataset)
        train_data_loader_dict[i] = dataloader
        test_data_loader_dict[i] = testloader
        # client_list.append(
        #     Client(i, train_data_loader_dict[i], len(dataset), opt, device, model, tb_writer=tb_writer,
        #            hyp=hyp, wandb=wandb))
        #



    return train_data_num, test_data_num, train_dataloader_global, test_dataloader_global, \
           train_data_num_dict, train_data_loader_dict, test_data_loader_dict, nc

