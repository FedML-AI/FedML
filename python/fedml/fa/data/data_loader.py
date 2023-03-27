import argparse
import logging
import os

from fedml.fa.data.fake_numeric_data.data_loader import generate_fake_data, load_partition_data_fake
from fedml.fa.data.twitter_Sentiment140.data_loader import download_twitter_Sentiment140, \
    load_partition_data_twitter_sentiment140


def fa_load_data(args):
    return load_synthetic_data(args)


def load_synthetic_data(args):
    dataset_name = args.dataset
    if dataset_name == "fake":
        data_cache_dir = os.path.join(args.data_cache_dir, "fake_numeric_data")
        if not os.path.exists(data_cache_dir):
            os.makedirs(data_cache_dir)
        print(f"---data_cache_dir={data_cache_dir}")
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
        # preprocess_twitter_data(path=path)
        (
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ) = load_partition_data_twitter_sentiment140(data_dir=path, client_num_in_total=int(args.client_num_in_total))

        dataset = [
            datasize,
            train_data_local_num_dict,
            local_data_dict,
        ]
        print(f"datasize, train_data_local_num_dict, local_data_dict,{dataset}")
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
