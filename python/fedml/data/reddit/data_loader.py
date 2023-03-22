import os
import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from torch.nn.utils.rnn import pad_sequence

from .datasets import Reddit_dataset
from .divide_data import DataPartitioner, select_dataset


from transformers import (AdamW, AlbertTokenizer, AutoConfig,
                            AutoModelWithLMHead, AutoTokenizer,
                            MobileBertForPreTraining)
                            
tokenizer = AlbertTokenizer.from_pretrained(
    "albert-base-v2", do_lower_case=True)

def collate_fn(examples):
    if tokenizer._pad_token is None:
        return (pad_sequence(examples, batch_first=True), None)
    return (pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id), None)


def load_partition_data_reddit(
    args,
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    n_proc_in_silo=0,
):

    from .nlp import load_and_cache_examples, mask_tokens
    from transformers import (AdamW, AlbertTokenizer, AutoConfig,
                              AutoModelWithLMHead, AutoTokenizer,
                              MobileBertForPreTraining)
    # tokenizer = AlbertTokenizer.from_pretrained(
        # 'albert-base-v2', do_lower_case=True)
    config = AutoConfig.from_pretrained(
        os.path.join(args.data_cache_dir, args.model+'-config.json'))
    # model = AutoModelWithLMHead.from_config(config)
    tokenizer = AlbertTokenizer.from_pretrained(
        args.model, do_lower_case=True)

    whole_train_dataset = load_and_cache_examples(
        args, tokenizer, evaluate=False)
    whole_test_dataset = load_and_cache_examples(
        args, tokenizer, evaluate=True)

    training_sets = DataPartitioner(
        data=whole_train_dataset, args=args, numOfClass=args.num_class)
    training_sets.partition_data_helper(
        num_clients=args.client_num_per_round, data_map_file=args.data_map_file)

    # testing_sets = DataPartitioner(
    #     data=test_dataset, args=args, numOfClass=args.num_class, isTest=True)
    # testing_sets.partition_data_helper(num_clients=num_executors)

    # class_num = len(np.unique(y_train))
    class_num = 0
    train_data_num = 0


    train_data_global = data.DataLoader(whole_train_dataset, 
        batch_size=batch_size, shuffle=True, pin_memory=True, 
        num_workers=args.num_loaders, drop_last=True, collate_fn=collate_fn)

    test_data_global = data.DataLoader(whole_test_dataset, 
        batch_size=batch_size, shuffle=True, pin_memory=True, 
        num_workers=args.num_loaders, drop_last=False, collate_fn=collate_fn)

    # train_data_global = data.DataLoader(whole_train_dataset, 
    #     batch_size=batch_size, shuffle=True, pin_memory=True, 
    #     timeout=60, num_workers=args.num_loaders, drop_last=True, collate_fn=collate_fn)

    # test_data_global = data.DataLoader(whole_test_dataset, 
    #     batch_size=batch_size, shuffle=True, pin_memory=True, 
    #     timeout=60, num_workers=args.num_loaders, drop_last=False, collate_fn=collate_fn)


    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global.dataset)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    filter_client_idx = 0
    num_clients = len(training_sets.partitions)
    for client_idx in range(num_clients):

        client_data = select_dataset(client_idx, training_sets,
                                        batch_size=args.batch_size, args=args,
                                        collate_fn=collate_fn)

        # data_loader = select_dataset(this_rank, testing_sets,
        #                              batch_size=args.test_bsz, args=args,
        #                              isTest=True, collate_fn=collate_fn
        #                              )

        local_data_num = len(client_data.dataset)
        if local_data_num < args.filter_less:
            continue
        else:
            train_data_num += local_data_num
            data_local_num_dict[filter_client_idx] = local_data_num
            logging.info(
                "filter_client_idx = %d, local_sample_number = %d" % (filter_client_idx, local_data_num)
            )

            logging.info(
                "filter_client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d"
                % (filter_client_idx, len(client_data), len([]))
            )
            train_data_local_dict[filter_client_idx] = client_data
            test_data_local_dict[filter_client_idx] = None
            filter_client_idx += 1

    logging.info(f"Total data num : {train_data_num}")
    args.client_num_in_total = filter_client_idx
    logging.info(f"Total clients: {num_clients}, \
        After filtering size less than {args.filter_less}, num of clients : {filter_client_idx}")

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
