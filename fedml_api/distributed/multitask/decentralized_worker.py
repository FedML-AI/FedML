import copy
import logging

import numpy as np
import torch
import wandb
from scipy.linalg import fractional_matrix_power
from torch import nn


class DecentralizedWorker(object):
    def __init__(self, worker_index, topology_manager, train_data_local_dict, test_data_local_dict,
                 train_data_local_num_dict, train_data_num, device, model, args):
        # topology management
        self.worker_index = worker_index
        self.in_neighbor_idx_list = topology_manager.get_in_neighbor_idx_list(worker_index)
        logging.info("in_neighbor_idx_list (index = %d) = %s" % (self.worker_index, str(self.in_neighbor_idx_list)))

        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_neighbor_result_received_dict = dict()
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_result_received_dict[neighbor_idx] = False

        # dataset
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[worker_index]
        self.local_sample_number = self.train_data_local_num_dict[worker_index]

        # model and optimization
        self.device = device
        self.args = args
        self.model = model
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.lambda_relationship = 0.1

        # initialize the task specific weights
        self.neighbor_task_specific_weight_dict = dict()
        task_specific_layer = self.model.task_specific_layer
        torch.nn.init.xavier_uniform_(task_specific_layer.weight)

        # correlation matrix
        num_of_neighors = len(self.in_neighbor_idx_list)
        self.corr_matrix_omega_default = torch.ones([num_of_neighors, num_of_neighors], device=self.device,
                                                    requires_grad=False) / num_of_neighors
        self.corr_matrix_omega = self.corr_matrix_omega_default

        # test result
        self.train_acc_dict = []
        self.train_loss_dict = []
        self.test_acc_dict = []
        self.test_loss_dict = []
        self.flag_neighbor_test_result_received_dict = dict()
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_test_result_received_dict[neighbor_idx] = False

        # self._register_hooks()

    def update_model(self, weights):
        self.model.load_state_dict(weights)

    def update_dataset(self, worker_index):
        self.worker_index = worker_index
        self.train_local = self.train_data_local_dict[worker_index]
        self.local_sample_number = self.train_data_local_num_dict[worker_index]

    def add_neighbor_local_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_neighbor_result_received_dict[index] = True
        # Note: Loss doesn't backprop through copy-based reshapes https://github.com/pytorch/xla/issues/870
        task_specific_weight = model_params['task_specific_layer.weight'].view(-1, )
        # logging.info("task_specific_weight = " + str(task_specific_weight))
        self.neighbor_task_specific_weight_dict[index] = task_specific_weight.to(self.device)

    def check_whether_all_receive(self):
        for neighbor_idx in self.in_neighbor_idx_list:
            if not self.flag_neighbor_result_received_dict[neighbor_idx]:
                return False
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_result_received_dict[neighbor_idx] = False
        return True

    def calculate_relationship_regularizer_with_trace(self):
        tensor_list = []
        # update local specific weights
        task_specific_weight = self.model.task_specific_layer.weight.view(-1, )
        self.neighbor_task_specific_weight_dict[self.worker_index] = task_specific_weight

        for neighbor_idx in self.in_neighbor_idx_list:
            logging.info("worker_index = %d, require_grad = %d" % (self.worker_index,
                                                                   self.neighbor_task_specific_weight_dict[neighbor_idx].requires_grad))
            tensor_list.append(self.neighbor_task_specific_weight_dict[neighbor_idx])

        weight_matrix = torch.stack(tensor_list, 0)
        trans_w = torch.transpose(weight_matrix, 0, 1)
        # (H, N_nb) * (N_nb, N_nb) * (N_nb, H)
        # logging.info("self.corr_matrix_omega = " + str(self.corr_matrix_omega))
        relationship_trace = torch.trace(torch.matmul(trans_w, torch.matmul(self.corr_matrix_omega, weight_matrix)))
        return relationship_trace

    def update_correlation_matrix(self):
        tensor_list = []
        task_specific_weight = self.model.task_specific_layer.weight.view(-1, )
        self.neighbor_task_specific_weight_dict[self.worker_index] = task_specific_weight

        for neighbor_idx in self.in_neighbor_idx_list:
            tensor_list.append(self.neighbor_task_specific_weight_dict[neighbor_idx].detach())

        weight_matrix = torch.stack(tensor_list, 0)
        trans_w = torch.transpose(weight_matrix, 0, 1)
        corr_trans = torch.matmul(weight_matrix, trans_w)
        # logging.info(corr_trans)
        corr_new_np = fractional_matrix_power(corr_trans.cpu(), 1 / 2)
        self.corr_matrix_omega = torch.from_numpy(corr_new_np / np.trace(corr_new_np)).float().to(self.device)
        self.corr_matrix_omega.requires_grad = False

    def aggregate(self):
        model_list = []
        training_num = 0

        for neighbor_idx in self.in_neighbor_idx_list:
            model_list.append((self.sample_num_dict[neighbor_idx], self.model_dict[neighbor_idx]))
            training_num += self.sample_num_dict[neighbor_idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.model.load_state_dict(averaged_params)

    def train(self):
        self.model.to(self.device)
        # change to train mode
        self.model.train()

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_local):

                # iterative step 1: update the theta, W_i
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                predict_loss = self.criterion(log_probs, labels)

                if self.args.is_mtl == 1:
                    relationship_trace = self.calculate_relationship_regularizer_with_trace()
                    total_loss = predict_loss + self.lambda_relationship * relationship_trace
                    total_loss.backward()
                else:
                    predict_loss.backward()

                self.optimizer.step()

                # iterative step 2: update relationship matrix omega
                if self.args.is_mtl == 1:
                    self.update_correlation_matrix()

                # batch_loss.append(total_loss.item())
                # if len(batch_loss) > 0:
                #     epoch_loss.append(sum(batch_loss) / len(batch_loss))
                #     logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.worker_index,
                #                                                     epoch, sum(epoch_loss) / len(epoch_loss)))
                break
            break

        weights = self.model.cpu().state_dict()
        return weights, self.local_sample_number

    def test_on_local_data(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            # train data
            train_tot_correct, train_num_sample, train_loss = self._infer(self.train_data_local_dict[self.worker_index])

            # test data
            test_tot_correct, test_num_sample, test_loss = self._infer(self.test_data_local_dict[self.worker_index])

            # test on training dataset
            train_acc = train_tot_correct / train_num_sample
            train_loss = train_loss / train_num_sample

            # test on test dataset
            test_acc = test_tot_correct / test_num_sample
            test_loss = test_loss / test_num_sample
            # logging.info("worker_index = %d, train_acc = %f, train_loss = %f, test_acc = %f, test_loss = %f" % (
            # self.worker_index, train_acc, train_loss, test_acc, test_loss))
            return train_acc, train_loss, test_acc, test_loss
        else:
            return None, None, None, None

    def _infer(self, test_data):
        self.model.eval()
        self.model.to(self.device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
        # logging.info("worker_index = %d, test_acc = %d, test_total = %d, test_loss = %d" % (self.worker_index, test_acc, test_total, test_loss))
        return test_acc, test_total, test_loss

    def _check_whether_all_test_result_receive(self):
        for neighbor_idx in self.in_neighbor_idx_list:
            if not self.flag_neighbor_test_result_received_dict[neighbor_idx]:
                return False
        for neighbor_idx in self.in_neighbor_idx_list:
            self.flag_neighbor_test_result_received_dict[neighbor_idx] = False
        return True

    def record_average_test_result(self, index, round_idx, train_acc, train_loss, test_acc, test_loss):
        self.train_acc_dict.append(train_acc)
        self.train_loss_dict.append(train_loss)
        self.test_acc_dict.append(test_acc)
        self.test_loss_dict.append(test_loss)
        self.flag_neighbor_test_result_received_dict[index] = True
        if self._check_whether_all_test_result_receive():
            logging.info("ROUND INDEX = %d" % round_idx)
            train_acc = sum(self.train_acc_dict) / len(self.train_acc_dict)
            train_loss = sum(self.train_loss_dict) / len(self.train_loss_dict)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            test_acc = sum(self.test_acc_dict) / len(self.test_acc_dict)
            test_loss = sum(self.test_loss_dict) / len(self.test_loss_dict)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)

            self.train_acc_dict.clear()
            self.train_loss_dict.clear()
            self.test_acc_dict.clear()
            self.test_loss_dict.clear()

    def _register_hooks(self):
        def hook_fn(m, i, o):
            logging.info(m)
            logging.info("------------Input Grad------------")

            for grad in i:
                try:
                    logging.info(grad.shape)
                except AttributeError:
                    logging.info("None found for Gradient")

            logging.info("------------Output Grad------------")
            for grad in o:
                try:
                    logging.info(grad.shape)
                except AttributeError:
                    logging.info("None found for Gradient")
            logging.info("\n")

        self.model.shared_NN_fc.register_backward_hook(hook_fn)
        self.model.task_specific_layer.register_backward_hook(hook_fn)
