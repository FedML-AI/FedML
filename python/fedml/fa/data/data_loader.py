import argparse
import logging
import os
from fedml.fa.constants import FA_TASK_HEAVY_HITTER_TRIEHH
from fedml.fa.data.fake_numeric_data.data_loader import generate_fake_data, load_partition_data_fake
from fedml.fa.data.self_defined_data.data_loader import load_partition_self_defined_data
from fedml.fa.data.twitter_Sentiment140.data_loader import download_twitter_Sentiment140, \
    load_partition_data_twitter_sentiment140, load_partition_data_twitter_sentiment140_heavy_hitter
from fedml.fa.data.twitter_Sentiment140.twitter_data_processing import preprocess_twitter_data, \
    preprocess_twitter_data_heavy_hitter


def fa_load_data(args):
    return load_synthetic_data(args)


def load_synthetic_data(args):
    dataset_name = args.dataset
    if dataset_name == "fake":
        data_cache_dir = os.path.join(args.data_cache_dir, "fake_numeric_data")
        if not os.path.exists(data_cache_dir):
            os.makedirs(data_cache_dir, exist_ok=True)
        generate_fake_data(data_cache_dir)
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ) = load_partition_data_fake(data_dir=data_cache_dir, client_num=int(args.client_num_in_total))

        dataset = [
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ]
        # print(f"datasize, train_data_local_num_dict, local_data_dict,{dataset}")
    elif dataset_name == "twitter":
        path = os.path.join(args.data_cache_dir, "twitter_Sentiment140")
        download_twitter_Sentiment140(data_cache_dir=path)
        if args.fa_task != FA_TASK_HEAVY_HITTER_TRIEHH:
            local_datasets = preprocess_twitter_data(path=path)
            (
                datasize,
                train_data_local_num_dict,
                local_data_dict,
            ) = load_partition_data_twitter_sentiment140(local_datasets, client_num_in_total=int(args.client_num_in_total))

            dataset = [
                datasize,
                train_data_local_num_dict,
                local_data_dict,
            ]
        else:
            local_datasets = preprocess_twitter_data_heavy_hitter(path=path)
            (
                datasize,
                train_data_local_num_dict,
                local_data_dict,
            ) = load_partition_data_twitter_sentiment140_heavy_hitter(local_datasets, int(args.client_num_in_total))
            dataset = [
                datasize,
                train_data_local_num_dict,
                local_data_dict,
            ]
    elif dataset_name == "self_defined":
        data_cache_dir = args.data_cache_dir
        if not os.path.exists(data_cache_dir):
            os.makedirs(data_cache_dir, exist_ok=True)
        if hasattr(args, "data_col_idx") and isinstance(args.data_col_idx, int) and args.data_col_idx >= 0:
            if hasattr(args, "seperator"):
                separator = args.seperator
            else:
                separator = ","  # default seperator = ","
            (
                datasize,
                train_data_local_num_dict,
                local_data_dict,
            ) = load_partition_self_defined_data(file_folder_path=data_cache_dir,
                                                 client_num=int(args.client_num_in_total),
                                                 data_col_idx=int(args.data_col_idx),
                                                 separator=separator)
            dataset = [
                datasize,
                train_data_local_num_dict,
                local_data_dict,
            ]

        else:
            raise Exception("illegal data column index (data_col_idx) in the data file(s)")
    else:
        raise "Not Implemented Error"
    return dataset


def load_synthetic_data_test():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--dataset", type=str, default="twitter")
    parser.add_argument("--data_cache_dir", type=str, default="data")
    parser.add_argument("--client_num_in_total", type=int, default=10)

    args, unknown = parser.parse_known_args()

    load_synthetic_data(args=args)


if __name__ == '__main__':
    # read_data(train_data_dir="fake_data")
    # download_twitter_Sentiment140("data")
    load_synthetic_data_test()
