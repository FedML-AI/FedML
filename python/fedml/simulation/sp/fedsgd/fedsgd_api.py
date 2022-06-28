import copy
import logging
import random

import numpy as np
import torch
import wandb

from .client import Client
from .my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
import logging

from ....utils.compression import compressors
from ....utils.model_utils import (
    average_named_params,
    get_average_weight
)


class FedSGDAPI(object):
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

        self.train_local_iter_dict = {}


        logging.info("model = {}".format(model))
        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model)
        else:
            # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model, device, args)
        self.model_trainer = model_trainer
        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            self.model_trainer,
        )

        if args.compression is None or self.args.compression == 'no':
            pass
        else:
            self.compressor = compressors[args.compression]()
            model_params = self.model_trainer.get_model_params()
            for k, v in model_params.items():
                self.compressor.update_shapes_dict(model_params[k], k)


    def _setup_clients(
            self,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")



    def get_train_batch_data(self, client_idx):
        try:
            # train_batch_data = self.train_local_iter.next()
            train_batch_data = self.train_local_iter_dict[client_idx].next()
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            # self.train_local_iter = iter(train_local)
            # train_batch_data = self.train_local_iter.next()
            self.train_local_iter_dict[client_idx] = iter(self.train_data_local_dict[client_idx])
            train_batch_data = self.train_local_iter_dict[client_idx].next()

        return train_batch_data


    def get_global_train_batch_data(self, client_idx):
        try:
            train_batch_data = self.train_global_iter.next()
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_global_iter = iter(self.train_global)
            train_batch_data = self.train_global_iter.next()

        return train_batch_data

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            g_locals = []
            bn_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # train on new dataset
                train_batch_data = self.get_train_batch_data(client_idx)
                # train_batch_data = self.get_global_train_batch_data(client_idx)
                compressed_grads, grad_indexes = client.train(copy.deepcopy(w_global), train_batch_data)
                bn_local = client.get_model_bn()
                # self.logging.info("local weights = " + str(w))
                g_locals.append([client.get_sample_number(), compressed_grads, grad_indexes])
                bn_locals.append(bn_local)

            # obtain global gradients
            averaged_g, averaged_bn_params = self._aggregate(g_locals, bn_locals)

            # update global weights
            # w_global = self._aggregate(w_locals)
            # self.model_trainer.set_model_params(w_global)
            self.model_trainer.clear_grad_params()
            # self.optimize_with_grad(grad_params)
            self.model_trainer.set_grad_params(averaged_g)
            self.model_trainer.update_model_with_grad()

            # self.model_trainer.set_model_bn(averaged_bn_params)
            w_global = self.model_trainer.get_model_params()

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self.test_on_server_for_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self.test_on_server_for_all_clients(round_idx)
                else:
                    self.test_on_server_for_all_clients(round_idx)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(
            range(test_data_num), min(num_samples, test_data_num)
        )
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size
        )
        self.val_global = sample_testset

    def uncompress_grad_params(self, grad_params, grad_indexes):
        if grad_indexes is not None:
            for k in grad_indexes.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k
                # ))
                grad_params[k] = self.compressor.unflatten(
                    self.compressor.decompress_new(grad_params[k], grad_indexes[k], k), k)
        elif self.args.compression is not None and self.args.compression != 'no':
            # TODO, add quantize here
            for k in grad_params.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k
                # ))
                grad_params[k] = self.compressor.decompress_new(grad_params[k])
        else:
            pass
        return grad_params


    def _aggregate(self, g_locals, bn_locals):

        sample_num_list = [item[0] for item in g_locals]

        new_g_locals = []
        for item in g_locals:
            # grad_params = self.uncompress_grad_params(
            #     grad_params, grad_indexes)
            grad_params = self.uncompress_grad_params(
                item[1], item[2])
            new_g_locals.append(grad_params)

        average_weights_dict_list = get_average_weight(sample_num_list)

        averaged_g = average_named_params(
            new_g_locals,
            average_weights_dict_list
        )

        averaged_bn_params = average_named_params(
            bn_locals,
            average_weights_dict_list
        )

        return averaged_g, averaged_bn_params




    def test_on_server_for_all_clients(self, round_idx):
        if (
            round_idx % self.args.frequency_of_the_test == 0
            or round_idx == self.args.comm_round - 1
        ):
            logging.info(
                "################test_on_server_for_all_clients : {}".format(round_idx)
            )
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            # for client_idx in range(self.args.client_num_in_total):
            #     # train data
            #     metrics = self.model_trainer.test(
            #         self.train_data_local_dict[client_idx], self.device, self.args
            #     )
            #     train_tot_correct, train_num_sample, train_loss = (
            #         metrics["test_correct"],
            #         metrics["test_total"],
            #         metrics["test_loss"],
            #     )
            #     train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            #     train_num_samples.append(copy.deepcopy(train_num_sample))
            #     train_losses.append(copy.deepcopy(train_loss))

            # test on training dataset
            # train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            # train_loss = sum(train_losses) / sum(train_num_samples)
            # if self.args.enable_wandb:
            #     wandb.log({"Train/Acc": train_acc, "round": round_idx})
            #     wandb.log({"Train/Loss": train_loss, "round": round_idx})
            # stats = {"training_acc": train_acc, "training_loss": train_loss}
            # logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.model_trainer.test(self.test_global, self.device, self.args)
            else:
                # metrics = self.model_trainer.test(self.val_global, self.device, self.args)
                metrics = self.model_trainer.test(self.test_global, self.device, self.args)

            test_tot_correct, test_num_sample, test_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            logging.info(stats)
