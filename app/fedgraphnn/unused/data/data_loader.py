import logging

import numpy as np
import torch

from moleculenet.data_loader import load_moleculenet
from recommender_systems.data_loader import load_recsys_data
from subgraphs.data_loader import load_subgraph_data


def load(args):
    return load_synthetic_data(args)


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def load_synthetic_data(args):
    dataset_name = args.dataset
    # check if the centralized training is enabled
    centralized = (
        True
        if (args.client_num_in_total == 1 and args.training_type != "cross_silo")
        else False
    )

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False
    

    if dataset_name in ["sider", "clintox", "bbbp", "esol", "freesolv", "herg", "lipo", "pcba", "tox21", "toxcast", "muv","hiv" , "qm7" , "qm8" , "qm9"]:
        #MoleculeNet datasets
        logging.info("load_data. dataset_name = %s" % dataset_name)
        dataset, num_cats, feat_dim = load_moleculenet(args, dataset_name)
        return dataset, num_cats, feat_dim
    elif dataset_name in ["ciao","epinions"]:
        #Recommender Systems
        logging.info("load_data. dataset_name = %s" % dataset_name)
        dataset, num_cats, feat_dim = load_recsys_data(args, dataset_name)
        return dataset, num_cats, feat_dim
    elif dataset_name in ["COLLAB","REDDIT-BINARY","REDDIT-MULTI-5K","REDDIT-MULTI-12K","IMDB-BINARY","IMDB-MULTI"]:
        #Social-networks
        logging.info("load_data. dataset_name = %s" % dataset_name)
        return None
    elif dataset_name in ["YAGO3-10", "wn18rr", "FB15k-237"]:
        #Subgraph level
        logging.info("load_data. dataset_name = %s" % dataset_name)
        dataset = load_subgraph_data(args, dataset_name)
        return dataset
    elif dataset_name in ["CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"]:
        #Federated Node classification
        logging.info("load_data. dataset_name = %s" % dataset_name)
        return None
    else:
        return None

    
    if centralized:
        train_data_local_num_dict = {
            0: sum(
                user_train_data_num
                for user_train_data_num in train_data_local_num_dict.values()
            )
        }
        train_data_local_dict = {
            0: [
                batch
                for cid in sorted(train_data_local_dict.keys())
                for batch in train_data_local_dict[cid]
            ]
        }
        test_data_local_dict = {
            0: [
                batch
                for cid in sorted(test_data_local_dict.keys())
                for batch in test_data_local_dict[cid]
            ]
        }
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {
            cid: combine_batches(train_data_local_dict[cid])
            for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {
            cid: combine_batches(test_data_local_dict[cid])
            for cid in test_data_local_dict.keys()
        }
        args.batch_size = args_batch_size

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
