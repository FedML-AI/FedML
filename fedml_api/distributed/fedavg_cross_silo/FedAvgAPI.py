from fedml_api.distributed.fedavg_cross_silo.DistWorker import DistWorker
import logging
from .FedAvgServerManager import FedAVGServerManager
from .ClientMasterManager import ClientMasterManager
from .ClientSlaveManager import ClientSlaveManager
from .DistAggregator import DistAggregator


from fedml_api.distributed.utils.gpu_mapping_cross_silo import corss_silo_mapping_processes_to_gpu_device_from_yaml_file


class FedAvgAPI:
    def __init__(self, silo_rank,
                 number_of_worker_silos,
                 silo_server_device,
                 comm,
                 model,
                 train_data_num,
                 train_data_global,
                 test_data_global,
                 train_data_local_num_dict,
                 train_data_local_dict,
                 test_data_local_dict,
                 args,
                 model_trainer=None,
                 preprocessed_sampling_lists=None):

        self.FedML_FedAvg_distributed(silo_rank,
                                      number_of_worker_silos,
                                      silo_server_device,
                                      comm,
                                      model,
                                      train_data_num,
                                      train_data_global,
                                      test_data_global,
                                      train_data_local_num_dict,
                                      train_data_local_dict,
                                      test_data_local_dict,
                                      args,
                                      model_trainer,
                                      preprocessed_sampling_lists)

    def get_dist_worker(self, args,
                        device,
                        silo_rank,
                        model,
                        train_data_num,
                        train_data_local_num_dict,
                        train_data_local_dict,
                        test_data_local_dict,
                        model_trainer):
        return DistWorker(args,
                          device,
                          silo_rank,
                          model,
                          train_data_num,
                          train_data_local_num_dict,
                          train_data_local_dict,
                          test_data_local_dict,
                          model_trainer)

    def get_dist_aggregator(self, args, device,
                            size,
                            model,
                            train_data_num,
                            train_data_global,
                            test_data_global,
                            train_data_local_dict,
                            test_data_local_dict,
                            train_data_local_num_dict,
                            model_trainer):
        return DistAggregator(args,
                              device,
                              size,
                              model,
                              train_data_num,
                              train_data_global,
                              test_data_global,
                              train_data_local_dict,
                              test_data_local_dict,
                              train_data_local_num_dict,
                              model_trainer)

    def get_server_manager(self, args,
                           dist_aggregator,
                           comm,
                           rank,
                           size,
                           backend,
                           is_preprocessed=False,
                           preprocessed_client_lists=None):
        return FedAVGServerManager(
            args,
            dist_aggregator,
            comm,
            rank,
            size,
            backend,
            is_preprocessed=is_preprocessed,
            preprocessed_client_lists=preprocessed_client_lists,
        )

    def get_clinet_manager_master(self, args, dist_worker, comm, silo_rank, size, backend):
        return ClientMasterManager(
            args, dist_worker, comm, silo_rank, size, backend)

    def get_clinet_manager_salve(self, args, dist_worker):
        return ClientSlaveManager(args, dist_worker)

    def FedML_FedAvg_distributed(
        self,
        silo_rank,
        number_of_worker_silos,
        silo_server_device,
        comm,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        args,
        model_trainer=None,
        preprocessed_sampling_lists=None,
    ):

        process_device = corss_silo_mapping_processes_to_gpu_device_from_yaml_file(
            args.silo_rank, args.silo_proc_rank, args.silo_proc_num, args.worker_silo_num, args.silo_gpu_mapping_file
        )

        if args.silo_proc_num == 0:
            assert silo_server_device == process_device, "GPU index mismatch between gpu_mapping and silo_gpu_mapping files"

        if not 'enable_cuda_rpc' in args:
            args.enable_cuda_rpc = False
        if silo_rank == 0:
            self.init_server(
                args,
                process_device,
                comm,
                silo_rank,
                number_of_worker_silos,
                model,
                train_data_num,
                train_data_global,
                test_data_global,
                train_data_local_dict,
                test_data_local_dict,
                train_data_local_num_dict,
                model_trainer,
                preprocessed_sampling_lists,
            )
        else:
            self.init_client(
                args,
                process_device,
                comm,
                silo_rank,
                number_of_worker_silos,
                model,
                train_data_num,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                model_trainer,
            )

    def init_server(
        self,
        args,
        device,
        comm,
        rank,
        size,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        model_trainer,
        preprocessed_sampling_lists=None,
    ):

        # start the distributed training
        backend = args.backend

        dist_aggregator = self.get_dist_aggregator(
            args,
            device,
            size,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            model_trainer
        )

        if preprocessed_sampling_lists is None:
            server_manager = self.get_server_manager(
                args, dist_aggregator, comm, rank, size, backend)
        else:
            server_manager = self.get_server_manager(
                args,
                dist_aggregator,
                comm,
                rank,
                size,
                backend,
                is_preprocessed=True,
                preprocessed_client_lists=preprocessed_sampling_lists,
            )
        server_manager.run()

    def init_client(
        self,
        args,
        device,
        comm,
        silo_rank,
        size,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer=None,
    ):
        backend = args.backend
        dist_worker = self.get_dist_worker(args,
                                           device,
                                           silo_rank,
                                           model,
                                           train_data_num,
                                           train_data_local_num_dict,
                                           train_data_local_dict,
                                           test_data_local_dict,
                                           model_trainer)
        if args.silo_proc_rank == 0:
            logging.info("Initiating Client Manager")
            client_manager = self.get_clinet_manager_master(
                args, dist_worker, comm, silo_rank, size, backend)
        else:
            logging.info("Initiating DDP worker")
            client_manager = self.get_clinet_manager_salve(args, dist_worker)
        logging.info("Running Client")
        client_manager.run()
