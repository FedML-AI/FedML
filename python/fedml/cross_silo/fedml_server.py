from fedml.core import ServerAggregator
from fedml.cross_silo.server.fedml_aggregator import FedMLAggregator
from fedml.cross_silo.server.fedml_server_manager import FedMLServerManager
from fedml.ml.aggregator.aggregator_creator import create_server_aggregator




class FedMLCrossSiloServer:
    def __init__(self, args, device, dataset, model, server_aggregator: ServerAggregator = None):
        self.server_manager = None
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
            # server_initializer.init_server(
            #     args,
            #     device,
            #     args.comm,
            #     args.rank,
            #     args.worker_num,
            #     model,
            #     train_data_num,
            #     train_data_global,
            #     test_data_global,
            #     train_data_local_dict,
            #     test_data_local_dict,
            #     train_data_local_num_dict,
            #     server_aggregator,
            # )
            if server_aggregator is None:
                server_aggregator = create_server_aggregator(model, args)
            server_aggregator.set_id(0)

            aggregator = FedMLAggregator(
                train_data_global,
                test_data_global,
                train_data_num,
                train_data_local_dict,
                test_data_local_dict,
                train_data_local_num_dict,
                args.worker_num,
                device,
                args,
                server_aggregator,
            )

            # start the distributed training
            backend = args.backend
            self.server_manager = FedMLServerManager(args, aggregator, args.comm, args.rank, args.worker_num, backend)
            self.server_manager.run()

        elif args.federated_optimizer == "LSA":
            from .lightsecagg.lsa_fedml_api import FedML_LSA_Horizontal

            FedML_LSA_Horizontal(
                args,
                args.rank,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=None,
                preprocessed_sampling_lists=None,
            )

        elif args.federated_optimizer == "SA":
            from .secagg.sa_fedml_api import FedML_SA_Horizontal

            FedML_SA_Horizontal(
                args,
                args.rank,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=None,
                preprocessed_sampling_lists=None,
            )
        else:
            raise Exception("Exception")

    def run(self):
        pass

    def set_model_params(self, model_params):
        self.server_manager.aggregator.set_global_model_params(model_params)

    def get_model_params(self):
        return self.server_manager.aggregator.get_global_model_params()
