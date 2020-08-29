import copy
import logging

import numpy as np
import wandb

from fedml_api.standalone.fedavg.client import Client


class FedAvgTrainer(object):
    def __init__(self, dataset, model, device, args):
        self.device = device
        self.args = args

        [train_data_num, test_data_num, train_data_global, test_data_global,
         data_local_num_dict, train_data_local_dict, test_data_local_dict, output_dim] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num = train_data_num
        self.test_data_num = test_data_num

        self.model_global = model
        self.model_global.train()

        self.client_list = []
        self.data_local_num_dict = data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.setup_clients(data_local_num_dict, train_data_local_dict, test_data_local_dict)

    def setup_clients(self, data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       data_local_num_dict[client_idx], self.args, self.device)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        return client_indexes

    def train(self):
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))

            self.model_global.train()
            w_locals, loss_locals = [], []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self.client_sampling(round_idx, self.args.client_num_in_total,
                                                  self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.data_local_num_dict[client_idx])

                # train on new dataset
                w, loss = client.train(net=copy.deepcopy(self.model_global).to(self.device))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                loss_locals.append(copy.deepcopy(loss))

            # update global weights
            w_glob = self.aggregate(w_locals)
            # logging.info("global weights = " + str(w_glob))

            # copy weight to net_glob
            self.model_global.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            logging.info('Round {:3d}, Average loss {:.3f}'.format(round_idx, loss_avg))

            self.local_test_on_all_clients(self.model_global, round_idx)

    def aggregate(self, w_locals):
        (num0, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / self.train_data_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def local_test_on_all_clients(self, model_global, round_idx):
        train_num_samples = []
        train_tot_corrects = []
        train_losses = []

        test_num_samples = []
        test_tot_corrects = []
        test_losses = []
        client = self.client_list[0]
        for client_idx in range(self.args.client_num_in_total):
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.data_local_num_dict[client_idx])
            # train data
            train_tot_correct, train_num_sample, train_loss = client.local_test(model_global, False)
            train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            train_num_samples.append(copy.deepcopy(train_num_sample))
            train_losses.append(copy.deepcopy(train_loss))

            # test data
            test_tot_correct, test_num_sample, test_loss = client.local_test(model_global, True)
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

        # test on training dataset
        train_acc = sum(train_tot_corrects) / sum(train_num_samples)
        train_loss = sum(train_losses) / sum(train_num_samples)
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        logging.info(stats)

        # test on test dataset
        test_acc = sum(test_tot_corrects) / sum(test_num_samples)
        test_loss = sum(test_losses) / sum(test_num_samples)
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info(stats)
