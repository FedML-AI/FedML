from fedml.core import ServerAggregator


class FedMLModelServingServer:
    def __init__(self, args, end_point_name, model_name, model_version,
                 inference_request=None, device=None, dataset=None, model=None,
                 server_aggregator: ServerAggregator = None):
        self.end_point_name = end_point_name
        self.model_name = model_name
        self.model_version = model_version
        self.inference_request = inference_request

        if args.federated_optimizer == "FedAvg":
            from fedml.cross_silo.server import server_initializer

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
