import logging
import time

import torch
import wandb
from torch import nn


class FedAVGAggregator(object):
    def __init__(self, train_global, test_global, all_train_data_num, client_num, device, model, args):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num
        self.client_num = client_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.train_acc_dict = dict()
        self.train_loss_dict = dict()
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.model, _ = self.init_model(model)

    def init_model(self, model):
        model_params = model.state_dict()
        # logging.info(model)
        return model, model_params

    def get_global_model_params(self):
        return self.model.state_dict()

    def add_local_trained_result(self, index, model_params, sample_num, train_acc, train_loss):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.train_acc_dict[index] = train_acc
        self.train_loss_dict[index] = train_loss
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        for idx in range(self.client_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / self.all_train_data_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.model.load_state_dict(averaged_params)

        # clear the memory cost
        model_list.clear()
        del model_list
        self.model_dict.clear()
        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def statistics(self, round_idx):
        # train acc
        train_acc_list = self.train_acc_dict.values()
        train_acc_avg = sum(train_acc_list) / len(train_acc_list)
        logging.info('Round {:3d}, Average Train Accuracy {:.3f}'.format(round_idx, train_acc_avg))
        wandb.log({"Train/AccTop1": train_acc_avg, "round": round_idx + 1})

        # train loss
        train_loss_list = self.train_loss_dict.values()
        train_loss_avg = sum(train_loss_list) / len(train_loss_list)
        logging.info('Round {:3d}, Average Train Loss {:.3f}'.format(round_idx, train_loss_avg))
        wandb.log({"Train/Loss": train_loss_avg, "round": round_idx + 1})

        # algorithms acc
        logging.info('Round {:3d}, Average Validation Accuracy {:.3f}'.format(round_idx, self.test_acc_avg))
        wandb.log({"Test/AccTop1": self.test_acc_avg, "round": round_idx + 1})

        # algorithms loss
        logging.info('Round {:3d}, Average Validation Loss {:.3f}'.format(round_idx, self.test_loss_avg))
        wandb.log({"Test/Loss": self.test_loss_avg, "round": round_idx + 1})

    def infer(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            start_time = time.time()
            self.model.eval()
            self.model.to(self.device)

            test_correct = 0.0
            test_loss = 0.0
            test_sample_number = 0.0
            test_data = self.test_global
            # loss
            criterion = nn.CrossEntropyLoss().to(self.device)
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_data):
                    x = x.to(self.device)
                    target = target.to(self.device)

                    pred = self.model(x)
                    loss = criterion(pred, target)
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                    test_correct += correct.item()
                    test_loss += loss.item() * target.size(0)
                    test_sample_number += target.size(0)
                logging.info("server algorithms. round_idx = %d, test_loss = %s" % (round_idx, test_loss))

            self.test_acc_avg = test_correct / test_sample_number
            self.test_acc_avg = self.test_acc_avg
            logging.info("self.test_acc_avg = " + str(self.test_acc_avg))
            self.test_loss_avg = test_loss

            end_time = time.time()
            logging.info("server_infer time cost: %d" % (end_time - start_time))
