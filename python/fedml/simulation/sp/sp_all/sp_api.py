import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer

from fedml.ml.trainer.my_model_trainer_classification_new import ModelTrainerCLS 
from fedml.simulation.mpi.mpi.FLTrainer import FLTrainer
from fedml.simulation.mpi.mpi.default_aggregator import DefaultServerAggregator

from fedml.ml.ml_message import MLMessage



class SPAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
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
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        logging.info("model = {}".format(model))
        # self.model_trainer = create_model_trainer(model, args)
        self.model_trainer = ModelTrainerCLS(model, args)
        self.model = model
        self.worker_num = args.client_num_per_round
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients()
        self.aggregator = DefaultServerAggregator(
            self.train_global,
            self.test_global,
            self.train_data_num_in_total,
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.train_data_local_num_dict,
            self.worker_num,
            self.device,
            self.args,
            self.model,
        )

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = FLTrainer(
                client_idx,
                self.train_data_local_dict,
                self.train_data_local_num_dict,
                self.test_data_local_dict,
                self.train_data_num_in_total,
                self.device,
                self.args,
                self.model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        # w_global = self.model_trainer.get_model_params()
        server_result = self.aggregator.get_init_server_result()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            # server_result = copy.deepcopy(server_result)
            client_indexes = self.aggregator.client_sampling(
                round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )
            server_result[MLMessage.SAMPLE_NUM_DICT] = dict([
                (client_index, self.train_data_local_num_dict[client_index]) for client_index in client_indexes
            ])
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_trainer(client_idx, copy.deepcopy(server_result))
                client.update_dataset(client_idx)
                # train on new dataset
                mlops.event("train", event_started=True, event_value="{}_{}".format(str(round_idx), str(idx)))
                client_result, local_sample_num = client.train(round_idx)
                mlops.event("train", event_started=False, event_value="{}_{}".format(str(round_idx), str(idx)))
                # self.logging.info("local weights = " + str(w))
                # w_locals.append((client.get_sample_number(), copy.deepcopy(client_result)))
                self.aggregator.add_local_trained_result(idx, copy.deepcopy(client_result), local_sample_num)

            # update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            server_result = self.aggregator.aggregate()
            mlops.event("agg", event_started=False, event_value=str(round_idx))
            self.aggregator.test_on_server_for_all_clients(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()
