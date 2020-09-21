import logging
import time

import torch
import wandb
from torch import nn


class FedNASAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num, client_num, model, device, args):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num
        self.client_num = client_num
        self.device = device
        self.args = args
        self.model = model
        self.model_dict = dict()
        self.arch_dict = dict()
        self.sample_num_dict = dict()
        self.train_acc_dict = dict()
        self.train_loss_dict = dict()
        self.train_acc_avg = 0.0
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.best_accuracy = 0
        self.best_accuracy_different_cnn_counts = dict()
        self.wandb_table = wandb.Table(columns=["Epoch", "Searched Architecture"])

    def get_model(self):
        return self.model

    def add_local_trained_result(self, index, model_params, arch_params, sample_num, train_acc, train_loss):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.arch_dict[index] = arch_params
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
        averaged_weights = self.__aggregate_weight()
        self.model.load_state_dict(averaged_weights)
        if self.args.stage == "search":
            averaged_alphas = self.__aggregate_alpha()
            self.__update_arch(averaged_alphas)
            return averaged_weights, averaged_alphas
        else:
            return averaged_weights

    def __update_arch(self, alphas):
        logging.info("update_arch. server.")
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    def __aggregate_weight(self):
        logging.info("################aggregate weights############")
        start_time = time.time()
        model_list = []
        for idx in range(self.client_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / self.all_train_data_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # clear the memory cost
        model_list.clear()
        del model_list
        self.model_dict.clear()
        end_time = time.time()
        logging.info("aggregate weights time cost: %d" % (end_time - start_time))
        return averaged_params

    def __aggregate_alpha(self):
        logging.info("################aggregate alphas############")
        start_time = time.time()
        alpha_list = []
        for idx in range(self.client_num):
            alpha_list.append((self.sample_num_dict[idx], self.arch_dict[idx]))

        (num0, averaged_alphas) = alpha_list[0]
        for index, alpha in enumerate(averaged_alphas):
            for i in range(0, len(alpha_list)):
                local_sample_number, local_alpha = alpha_list[i]
                w = local_sample_number / self.all_train_data_num
                if i == 0:
                    alpha = local_alpha[index] * w
                else:
                    alpha += local_alpha[index] * w
        end_time = time.time()
        logging.info("aggregate alphas time cost: %d" % (end_time - start_time))
        return averaged_alphas

    def statistics(self, round_idx):
        # train acc
        train_acc_list = self.train_acc_dict.values()
        self.train_acc_avg = sum(train_acc_list) / len(train_acc_list)
        logging.info('Round {:3d}, Average Train Accuracy {:.3f}'.format(round_idx, self.train_acc_avg))
        wandb.log({"Train Accuracy": self.train_acc_avg, "Round": round_idx})
        # train loss
        train_loss_list = self.train_loss_dict.values()
        train_loss_avg = sum(train_loss_list) / len(train_loss_list)
        logging.info('Round {:3d}, Average Train Loss {:.3f}'.format(round_idx, train_loss_avg))
        wandb.log({"Train Loss": train_loss_avg, "Round": round_idx})

        # test acc
        logging.info('Round {:3d}, Average Validation Accuracy {:.3f}'.format(round_idx, self.test_acc_avg))
        wandb.log({"Validation Accuracy": self.test_acc_avg, "Round": round_idx})
        # test loss
        logging.info('Round {:3d}, Average Validation Loss {:.3f}'.format(round_idx, self.test_loss_avg))
        wandb.log({"Validation Loss": self.test_loss_avg, "Round": round_idx})

        logging.info("search_train_valid_acc_gap %f" % (self.train_acc_avg - self.test_loss_avg))
        wandb.log({"search_train_valid_acc_gap": self.train_acc_avg - self.test_loss_avg, "Round": round_idx})

    def infer(self, round_idx):
        self.model.eval()
        self.model.to(self.device)
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            start_time = time.time()
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
                    if self.args.stage == "train":
                        loss = criterion(pred[0], target)
                        _, predicted = torch.max(pred[0], 1)
                    else:
                        loss = criterion(pred, target)
                        _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                    test_correct += correct.item()
                    test_loss += loss.item() * target.size(0)
                    test_sample_number += target.size(0)
                logging.info("server test. round_idx = %d, test_loss = %s" % (round_idx, test_loss))

            self.test_acc_avg = test_correct / test_sample_number
            self.test_loss_avg = test_loss

            end_time = time.time()
            logging.info("server_infer time cost: %d" % (end_time - start_time))

    def record_model_global_architecture(self, round_idx):
        # save the structure
        genotype, normal_cnn_count, reduce_cnn_count = self.model.genotype()
        cnn_count = normal_cnn_count + reduce_cnn_count
        wandb.log({"cnn_count": cnn_count, "Round": round_idx})

        logging.info("(n:%d,r:%d)" % (normal_cnn_count, reduce_cnn_count))
        logging.info('genotype = %s', genotype)
        wandb.log({"genotype": str(genotype), "round_idx": round_idx})

        self.wandb_table.add_data(str(round_idx), str(genotype))
        wandb.log({"Searched Architecture": self.wandb_table})

        # save the cnn architecture according to the CNN count
        cnn_count = normal_cnn_count * 10 + reduce_cnn_count
        wandb.log({"searching_cnn_count(%s)" % cnn_count: self.test_acc_avg, "epoch": round_idx})
        if cnn_count not in self.best_accuracy_different_cnn_counts.keys():
            self.best_accuracy_different_cnn_counts[cnn_count] = self.test_acc_avg
            summary_key_cnn_structure = "best_acc_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
            wandb.run.summary[summary_key_cnn_structure] = self.test_acc_avg

            summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
            wandb.run.summary[summary_key_best_cnn_structure] = round_idx
        else:
            if self.test_acc_avg > self.best_accuracy_different_cnn_counts[cnn_count]:
                self.best_accuracy_different_cnn_counts[cnn_count] = self.test_acc_avg
                summary_key_cnn_structure = "best_acc_for_cnn_structure(n:%d,r:%d)" % (
                    normal_cnn_count, reduce_cnn_count)
                wandb.run.summary[summary_key_cnn_structure] = self.test_acc_avg

                summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (
                    normal_cnn_count, reduce_cnn_count)
                wandb.run.summary[summary_key_best_cnn_structure] = round_idx

        if self.test_acc_avg > self.best_accuracy:
            self.best_accuracy = self.test_acc_avg
            wandb.run.summary["best_valid_accuracy"] = self.best_accuracy
            wandb.run.summary["epoch_of_best_accuracy"] = round_idx
