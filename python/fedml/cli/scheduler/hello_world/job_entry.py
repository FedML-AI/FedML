# import torch
#
import logging
import time

from fedml import FedMLRunner, mlops, constants
import fedml

# from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist
#
#
# def load_data(args):
#     download_mnist(args.data_cache_dir)
#     fedml.logging.info("load_data. dataset_name = %s" % args.dataset)
#
#     """
#     Please read through the data loader at to see how to customize the dataset for FedML framework.
#     """
#     (
#         client_num,
#         train_data_num,
#         test_data_num,
#         train_data_global,
#         test_data_global,
#         train_data_local_num_dict,
#         train_data_local_dict,
#         test_data_local_dict,
#         class_num,
#     ) = load_partition_data_mnist(
#         args,
#         args.batch_size,
#         train_path=args.data_cache_dir + "/MNIST/train",
#         test_path=args.data_cache_dir + "/MNIST/test",
#     )
#     """
#     For shallow NN or linear models,
#     we uniformly sample a fraction of clients each round (as the original FedAvg paper)
#     """
#     args.client_num_in_total = client_num
#     dataset = [
#         train_data_num,
#         test_data_num,
#         train_data_global,
#         test_data_global,
#         train_data_local_num_dict,
#         train_data_local_dict,
#         test_data_local_dict,
#         class_num,
#     ]
#     return dataset, class_num
#
#
# class LogisticRegression(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         import torch
#         outputs = torch.sigmoid(self.linear(x))
#         return outputs


if __name__ == "__main__":

    # Init logs before the program starts to log.
    mlops.log_print_init()

    # Use print or logging.info to print your logs, which will be uploaded to MLOps and can be showed in the logs page.
    print("Hello world. Here is the Falcon platform.")
    # logging.info("Hello world. Here is the Falcon platform.")

    time.sleep(10)

    # Cleanup logs when the program will be ended.
    mlops.log_print_cleanup()

    #
    # # init device
    # device = fedml.device.get_device(args)
    #
    # # load data
    # dataset, output_dim = load_data(args)
    #
    # # load model (the size of MNIST image is 28 x 28)
    # model = LogisticRegression(28 * 28, output_dim)
    #
    # # start training
    # fedml_runner = FedMLRunner(args, device, dataset, model)
    # fedml_runner.run()
