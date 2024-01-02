from .server import server_initializer
from ..core import ServerAggregator


class FedMLCrossCloudServer:
    def __init__(self, args, device, dataset, model, server_aggregator: ServerAggregator = None):
        if args.federated_optimizer == "FedAvg":
            [
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ] = dataset
            server_initializer.init_server(
                args,
                device,
                args.comm,
                args.rank,
                args.worker_num,
                model,
                train_data_num,
                train_data_global,
                test_data_global,
                train_data_local_dict,
                test_data_local_dict,
                train_data_local_num_dict,
                server_aggregator,
            )

        else:
            raise Exception("Exception")

    def run(self):
        pass
