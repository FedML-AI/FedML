import argparse
import logging
import time
import os
import sys

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
import data.raw_data_loader.news_20.data_loader
import data.raw_data_loader.AGNews.data_loader
import data.raw_data_loader.CNN_Dailymail.data_loader
import data.raw_data_loader.CornellMovieDialogue.data_loader
import data.raw_data_loader.SemEval2010Task8.data_loader
import data.raw_data_loader.Sentiment140.data_loader
import data.raw_data_loader.SQuAD_1_1.data_loader
import data.raw_data_loader.SST_2.data_loader
import data.raw_data_loader.W_NUT.data_loader
import data.raw_data_loader.wikiner.data_loader
import data.raw_data_loader.WMT.data_loader


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by running data loader
    """

    parser.add_argument(
        "--dataset",
        type=str,
        default="20news",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../data/text_classification/20Newsgroups",
        help="data directory",
    )

    parser.add_argument(
        "--h5_file_path", type=str, default="./20news_data.h5", help="h5 data file path"
    )

    args = parser.parse_args()
    return args


def test_raw_data_loader(args, dataset_name):
    data_loader = None
    if dataset_name == "20news":
        data_loader = data.raw_data_loader.news_20.data_loader.RawDataLoader(
            args.data_dir
        )
    elif dataset_name == "agnews":
        data_loader = data.raw_data_loader.AGNews.data_loader.RawDataLoader(
            args.data_dir
        )
    elif dataset_name == "cnn_dailymail":
        data_loader = data.raw_data_loader.CNN_Dailymail.data_loader.RawDataLoader(
            args.data_dir
        )
    elif dataset_name == "cornell_movie_dialogue":
        data_loader = (
            data.raw_data_loader.CornellMovieDialogue.data_loader.RawDataLoader(
                args.data_dir
            )
        )
    elif dataset_name == "semeval_2010_task8":
        data_loader = data.raw_data_loader.SemEval2010Task8.data_loader.RawDataLoader(
            args.data_dir
        )
    elif dataset_name == "sentiment140":
        data_loader = data.raw_data_loader.Sentiment140.data_loader.RawDataLoader(
            args.data_dir
        )
    elif dataset_name == "squad_1.1":
        data_loader = data.raw_data_loader.SQuAD_1_1.data_loader.RawDataLoader(
            args.data_dir
        )
    elif dataset_name == "sst_2":
        data_loader = data.raw_data_loader.SST_2.data_loader.RawDataLoader(
            args.data_dir
        )
    elif dataset_name == "w_nut":
        data_loader = data.raw_data_loader.W_NUT.data_loader.RawDataLoader(
            args.data_dir
        )
    elif dataset_name == "wikiner":
        data_loader = data.raw_data_loader.wikiner.data_loader.RawDataLoader(
            args.data_dir
        )
    elif dataset_name.startswith("wmt"):
        data_loader = data.raw_data_loader.WMT.data_loader.RawDataLoader(
            args.data_dir.split(",")
        )
    else:
        raise Exception("No such dataset!!")
    logger.info("Start loading data from files")
    start = time.time()
    data_loader.load_data()
    end = time.time()
    logger.info("Finish loading data from files, it takes %f sec" % (end - start))

    logger.info("Start generating h5 data file")
    start = time.time()
    data_loader.generate_h5_file(args.h5_file_path)
    end = time.time()
    logger.info("Finish generating h5 data file, it takes %f sec" % (end - start))


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    args = add_args(argparse.ArgumentParser(description="DataLoader-Test"))
    logger.info("Test %s DataLoader" % args.dataset)
    logger.info(args)

    logger.info("Start testing RawDataLoader")
    start = time.time()
    test_raw_data_loader(args, args.dataset)
    end = time.time()
    logger.info("Finish testing RawDataLoader, it takes %f sec" % (end - start))
