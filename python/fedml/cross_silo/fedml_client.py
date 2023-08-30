from fedml.constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL, FEDML_CROSS_SILO_SCENARIO_HORIZONTAL
from .client import client_initializer
from ..core import ClientTrainer


class FedMLCrossSiloClient:
    def __init__(self, args, device, dataset, model, model_trainer: ClientTrainer = None):
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
            # client_initializer.init_client(
            #     args,
            #     device,
            #     args.comm,
            #     args.rank,
            #     args.worker_num,
            #     model,
            #     train_data_num,
            #     train_data_local_num_dict,
            #     train_data_local_dict,
            #     test_data_local_dict,
            #     model_trainer,
            # )


            backend = args.backend

            trainer_dist_adapter = client_initializer.get_trainer_dist_adapter(
                args,
                device,
                args.rank,
                model,
                train_data_num,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                model_trainer,
            )
            if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
                if args.proc_rank_in_silo == 0:

                    self.client_manager = client_initializer.get_client_manager_master(
                        args, trainer_dist_adapter, args.comm, args.rank, args.worker_num, backend
                    )

                else:
                    self.client_manager = client_initializer.get_client_manager_salve(args, trainer_dist_adapter)

            elif args.scenario == FEDML_CROSS_SILO_SCENARIO_HORIZONTAL:

                self.client_manager = client_initializer.get_client_manager_master(args, trainer_dist_adapter, args.comm, args.rank, args.worker_num, backend)

            else:
                raise Exception("we do not support {}. Please check whether this is typo.".format(args.scenario))

            self.client_manager.run()

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
                model_trainer=model_trainer,
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
        self.client_manager.trainer_dist_adapter.update_model(model_params)

    def get_model_params(self):
        self.client_manager.trainer_dist_adapter.trainer.trainer.get_model_params()
