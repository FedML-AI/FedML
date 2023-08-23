from .client import client_initializer
from ..base_frame.client_analyzer import FAClientAnalyzer


class FACrossSiloClient:
    def __init__(self, args, dataset, client_analyzer: FAClientAnalyzer = None):
        [
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
        ] = dataset
        client_initializer.init_client(
            args,
            args.comm,
            args.rank,
            args.worker_num,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            client_analyzer,
        )

    def run(self):
        pass
