import copy
import logging

import pandas as pd

import socket

import numpy as np
import wandb

from fedml_api.standalone.fedavg.client import Client


class FedAvgTrainer(object):
    def __init__(self, dataset, model, device, args):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.model_global = model
        self.model_global.train()

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)

    def setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def client_sampling(self):
        # setup client
        hostname = socket.gethostname()
        client_socket = socket.socket()
        client_socket.connect((hostname, 8999))

        # get response
        response = client_socket.recv(1024).decode().split(',')

        # close connection
        client_socket.close()

        client_indexes = [int(i) for i in response[0].split(' ')]
        local_itr = [int(i) for i in response[1].split(' ')][0]
        radio_res = [float(i) for i in response[2].split(' ')]

        return client_indexes, local_itr, radio_res

    def train(self):
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))

            self.model_global.train()
            w_locals, loss_locals = [], []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes, local_itr, radio_res = self.client_sampling()
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                w, loss, time_interval = client.train(net=copy.deepcopy(self.model_global).to(self.device), local_iteration = local_itr)

                time_cost = time_interval + ((sum(w["linear.weight"]) + sum(w["linear.bias"])) / radio_res[idx])

                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                loss_locals.append(copy.deepcopy(loss))
                logging.info('Client {:3d}, loss {:.3f}'.format(client_idx, loss))

            # update global weights
            w_glob = self.aggregate(w_locals)
            # logging.info("global weights = " + str(w_glob))

            # copy weight to net_glob
            self.model_global.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            logging.info('Round {:3d}, Average loss {:.3f}'.format(round_idx, loss_avg))

            if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
                self.local_test_on_all_clients(self.model_global, round_idx)

    def aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def local_test_on_all_clients(self, model_global, round_idx):
        logging.info("################local_test_on_all_clients : {}".format(round_idx))
        train_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }
        
        test_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }

        client = self.client_list[0]
        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(model_global, False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(model_global, True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            if self.args.dataset == "stackoverflow_lr":
                train_metrics['precisions'].append(copy.deepcopy(train_local_metrics['test_precision']))
                train_metrics['recalls'].append(copy.deepcopy(train_local_metrics['test_recall']))
                test_metrics['precisions'].append(copy.deepcopy(test_local_metrics['test_precision']))
                test_metrics['recalls'].append(copy.deepcopy(test_local_metrics['test_recall']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
        train_precision = sum(train_metrics['precisions']) / sum(train_metrics['num_samples'])
        train_recall = sum(train_metrics['recalls']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        test_precision = sum(test_metrics['precisions']) / sum(test_metrics['num_samples'])
        test_recall = sum(test_metrics['recalls']) / sum(test_metrics['num_samples'])

        if self.args.dataset == "stackoverflow_lr":
            stats = {'training_acc': train_acc, 'training_precision': train_precision, 'training_recall': train_recall, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Pre": train_precision, "round": round_idx})
            wandb.log({"Train/Rec": train_recall, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_precision': test_precision, 'test_recall': test_recall, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_precision, "round": round_idx})
            wandb.log({"Test/Rec": test_recall, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)

        else:
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)
