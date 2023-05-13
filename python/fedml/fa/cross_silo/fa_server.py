from fedml.fa.base_frame.server_aggregator import FAServerAggregator
from fedml.fa.cross_silo.server.server_initializer import init_server


class FACrossSiloServer:
    def __init__(self, args, dataset, server_aggregator: FAServerAggregator = None):
        [
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
        ] = dataset
        init_server(
            args,
            args.comm,
            args.rank,
            args.worker_num,
            train_data_num,
            train_data_local_dict,
            train_data_local_num_dict,
            server_aggregator,
        )

    def run(self):
        pass
