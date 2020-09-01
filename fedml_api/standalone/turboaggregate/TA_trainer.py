import copy
import logging

import torch
import wandb
from torch import nn

from fedml_api.standalone.turboaggregate.TA_client import TA_Client


class TurboAggregateTrainer(object):
    def __init__(self, dataset, model, device, args):
        self.device = device
        self.args = args

        [train_data_num, test_data_num, train_data_global, test_data_global,
         data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.class_num = class_num
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num = train_data_num
        self.test_data_num = test_data_num

        self.model_global = model
        self.model_global.train()

        self.client_list = []
        self.setup_clients(data_local_num_dict, train_data_local_dict, test_data_local_dict)

    def setup_clients(self, data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_number):
            c = TA_Client(train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                          data_local_num_dict[client_idx], self.args, self.device)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        for round_idx in range(self.args.comm_round):
            logging.info("Communication round : {}".format(round_idx))

            self.model_global.train()
            w_locals, loss_locals = [], []
            for idx, client in enumerate(self.client_list):
                w, loss = client.train(net=copy.deepcopy(self.model_global).to(self.device))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                loss_locals.append(copy.deepcopy(loss))

            #########################################
            # Turbo-Aggregate Protocol Starts HERE. #
            #########################################

            # create the network topology
            self.TA_topology_vanilla()

            #######################################
            # Turbo-Aggregate Protocol Ends HERE. #
            #######################################

            # update global weights
            w_glob = self.aggregate(w_locals)
            # logging.info("global weights = " + str(w_glob))

            # copy weight to net_glob
            self.model_global.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            logging.info('Round {:3d}, Average loss {:.3f}'.format(round_idx, loss_avg))

            self.local_test(self.model_global, round_idx)

    def aggregate(self, w_locals):
        logging.info("################aggregate: %d" % len(w_locals))
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

    def TA_topology_vanilla(self):
        # logging.info("################aggregate: %d" % len(w_locals))

        # N = self.args.client_number
        # n_users_layer = np.ceil(np.log(N)).astype(int)
        # n_layer = np.ceil(float(N) / float(n_users_layer)).astype(int)

        # Set List of send_to, send_from

        # Initialize the buffer of clients
        pass

    def local_test(self, model_global, round_idx):
        self.local_test_on_training_data(model_global, round_idx)
        self.local_test_on_test_data(model_global, round_idx)

    def local_test_on_training_data(self, model_global, round_idx):
        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.client_list:
            tot_correct, num_sample, loss = c.local_test(model_global, False)

            tot_corrects.append(copy.deepcopy(tot_correct))
            num_samples.append(copy.deepcopy(num_sample))
            losses.append(copy.deepcopy(loss))

        train_acc = sum(tot_corrects) / sum(num_samples)
        train_loss = sum(losses) / sum(num_samples)

        wandb.log({"Train/AccTop1": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})

        stats = {'training_acc': train_acc, 'training_loss': train_loss, 'num_samples': num_samples}
        logging.info(stats)

    def local_test_on_test_data(self, model_global, round_idx):
        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.client_list:
            tot_correct, num_sample, loss = c.local_test(model_global, True)

            tot_corrects.append(copy.deepcopy(tot_correct))
            num_samples.append(copy.deepcopy(num_sample))
            losses.append(copy.deepcopy(loss))

        test_acc = sum(tot_corrects) / sum(num_samples)
        test_loss = sum(losses) / sum(num_samples)

        wandb.log({"Test/AccTop1": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})

        stats = {'test_acc': test_acc, 'test_loss': test_loss, 'num_samples': num_samples}
        logging.info(stats)

    def global_test(self):
        logging.info("################global_test")
        acc_train, num_sample, loss_train = self.test_using_global_dataset(self.model_global, self.train_global,
                                                                           self.device)
        acc_train = acc_train / num_sample

        acc_test, num_sample, loss_test = self.test_using_global_dataset(self.model_global, self.test_global,
                                                                         self.device)
        acc_test = acc_test / num_sample

        logging.info("Global Training Accuracy: {:.2f}".format(acc_train))
        logging.info("Global Testing Accuracy: {:.2f}".format(acc_test))
        wandb.log({"Global Training Accuracy": acc_train})
        wandb.log({"Global Testing Accuracy": acc_test})

    def test_using_global_dataset(self, model_global, global_test_data, device):
        model_global.eval()
        model_global.to(device)
        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(global_test_data):
                x = x.to(device)
                target = target.to(device)

                pred = model_global(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        return test_acc, test_total, test_loss
