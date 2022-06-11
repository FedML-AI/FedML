# import argparse
# import logging
# import time
# import os
# import sys

# # add the FedML root directory to the python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
# import data_preprocessing.news_20.data_loader
# import data_preprocessing.AGNews.data_loader
# import data_preprocessing.CNN_Dailymail.data_loader
# import data_preprocessing.CornellMovieDialogue.data_loader
# import data_preprocessing.SemEval2010Task8.data_loader
# import data_preprocessing.Sentiment140.data_loader
# import data_preprocessing.SQuAD_1_1.data_loader
# import data_preprocessing.SST_2.data_loader
# import data_preprocessing.W_NUT.data_loader
# import data_preprocessing.wikiner.data_loader
# import data_preprocessing.WMT.data_loader

# from data_preprocessing.base.partition import uniform_partition
# from data_preprocessing.base.utils import generate_h5_from_dict

# def add_args(parser):
#     """
#     parser : argparse.ArgumentParser
#     return a parser added with args required by running data loader
#     """

#     parser.add_argument('--dataset', type=str, default='20news', metavar='N',
#                         help='dataset used for training')

#     parser.add_argument('--data_dir', type=str, default='../../data/text_classification/20Newsgroups',
#                         help='data directory')

#     parser.add_argument('--partition_method', type=str, default='uniform', metavar='N',
#                         help='how to partition the dataset on local workers')

#     parser.add_argument('--batch_size', type=int, default=32, metavar='N',
#                         help='input batch size for training (default: 32)')

#     parser.add_argument('--client_num', type=int, default=100, metavar='NN',
#                         help='number of workers in a distributed cluster')

#     args = parser.parse_args()
#     return args


# def test_raw_data_loader(args, dataset_name):
#     data_loader = None
#     if dataset_name == "20news":
#         data_loader = data_preprocessing.news_20.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name == "agnews":
#         data_loader = data_preprocessing.AGNews.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name == "cnn_dailymail":
#         data_loader = data_preprocessing.CNN_Dailymail.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name == "cornell_movie_dialogue":
#         data_loader = data_preprocessing.CornellMovieDialogue.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name == "semeval_2010_task8":
#         data_loader = data_preprocessing.SemEval2010Task8.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name == "sentiment_140":
#         data_loader = data_preprocessing.Sentiment140.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name == "squad_1.1":
#         data_loader = data_preprocessing.SQuAD_1_1.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name == "sst_2":
#         data_loader = data_preprocessing.SST_2.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name == "w_nut":
#         data_loader = data_preprocessing.W_NUT.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name == "wikiner":
#         data_loader = data_preprocessing.wikiner.data_loader.RawDataLoader(args.data_dir)
#     elif dataset_name.startswith("wmt"):
#         data_loader = data_preprocessing.WMT.data_loader.RawDataLoader(args.data_dir.split(","))
#     else:
#         raise Exception("No such dataset!!")
#     results = data_loader.data_loader()
#     return data_loader, results


# def test_partition_method(args, partition_name, data_loader, results):
#     partition_dict = None
#     if partition_name == "uniform":
#         if "test_index_list" in results["attributes"]:
#             uniform_partition_dict = uniform_partition(results["attributes"]["train_index_list"],
#                                                results["attributes"]["test_index_list"], args.client_num)
#         else:
#             uniform_partition_dict = uniform_partition(results["attributes"]["index_list"], n_clients=args.client_num)
#         partition_dict = {"uniform": uniform_partition_dict}
#     elif partition_name == "nature":
#         if data_loader.nature_partition is not None and callable(data_loader.nature_partition):
#             nature_partition_dict = data_loader.nature_partition(results["attributes"])
#             partition_dict = {"nature": nature_partition_dict}
#         else:
#             raise Exception("No such nature partition method!!")
#     else:
#         raise Exception("No such partition method!!")
#     return partition_dict


# def test_client_data_loader(args, dataset_name, data_path, partition_path):
#     server_data_loader = None
#     client_data_loader = None
#     if dataset_name == "20news":
#         server_data_loader = data_preprocessing.news_20.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                           partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.news_20.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name == "agnews":
#         server_data_loader = data_preprocessing.AGNews.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.AGNews.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name == "cnn_dailymail":
#         server_data_loader = data_preprocessing.CNN_Dailymail.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.CNN_Dailymail.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name == "cornell_movie_dialogue":
#         server_data_loader = data_preprocessing.CornellMovieDialogue.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.CornellMovieDialogue.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name == "semeval_2010_task8":
#         server_data_loader = data_preprocessing.SemEval2010Task8.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.SemEval2010Task8.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name == "sentiment_140":
#         server_data_loader = data_preprocessing.Sentiment140.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.Sentiment140.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name == "squad_1.1":
#         server_data_loader = data_preprocessing.SQuAD_1_1.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.SQuAD_1_1.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name == "sst_2":
#         server_data_loader = data_preprocessing.SST_2.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.SST_2.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name == "w_nut":
#         server_data_loader = data_preprocessing.W_NUT.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.W_NUT.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name == "wikiner":
#         server_data_loader = data_preprocessing.wikiner.data_loader.ClientDataLoader(data_path, partition_path, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.wikiner.data_loader.ClientDataLoader(data_path, partition_path, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     elif dataset_name.startswith("wmt"):
#         data_file_paths = args.data_dir.split(",")
#         language_pair = (data_file_paths[0].split(".")[-1], data_file_paths[1].split(".")[-1])
#         server_data_loader = data_preprocessing.WMT.data_loader.ClientDataLoader(data_path, partition_path, language_pair, client_idx=None,
#                                                                  partition_method=args.partition_method)
#         client_data_loader = data_preprocessing.WMT.data_loader.ClientDataLoader(data_path, partition_path, language_pair, client_idx=0,
#                                                                  partition_method=args.partition_method)
#     else:
#         raise Exception("No such dataset!!")
#     train_batch_data_list = server_data_loader.get_train_batch_data(batch_size=args.batch_size)
#     test_batch_data_list = client_data_loader.get_test_batch_data(batch_size=args.batch_size)
#     train_batch_data_list = server_data_loader.get_train_batch_data(batch_size=args.batch_size)
#     test_batch_data_list = client_data_loader.get_test_batch_data(batch_size=args.batch_size)


# if __name__ == "__main__":
#     logging.basicConfig()
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)

#     args = add_args(argparse.ArgumentParser(description='DataLoader-Test'))
#     logger.info("Test %s DataLoader" % args.dataset)
#     logger.info(args)

#     logger.info("Start testing RawDataLoader")
#     start = time.time()
#     data_loader, results = test_raw_data_loader(args, args.dataset)
#     end = time.time()
#     logger.info("Finish testing RawDataLoader, it takes %f sec" % (end - start))

#     logger.info("Start testing partition method")
#     start = time.time()
#     partition_dict = test_partition_method(args, args.partition_method, data_loader, results)
#     end = time.time()
#     logger.info("Finish testing partition method, it takes %f sec" % (end - start))

#     data_path = args.dataset + "_data.h5"
#     partition_path = args.dataset + "_partition.h5"
#     generate_h5_from_dict(data_path, results)
#     generate_h5_from_dict(partition_path, partition_dict)

#     logger.info("Start testing ClientDataLoader")
#     start = time.time()
#     test_client_data_loader(args, args.dataset, data_path, partition_path)
#     end = time.time()
#     logger.info("Finish testing ClientDataLoader, it takes %f sec" % (end - start))
