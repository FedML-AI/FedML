import copy
import time

import numpy as np
import torch
import wandb

from darts import genotypes, utils
from darts.model import NetworkCIFAR
from darts.model_search import Network
from data_loader import get_dataloader
from torch import nn

import darts.genotypes

from client import Client


class FedNASTrainer(object):
    def __init__(self, net_dataidx_map, train_global, test_global, device, args, n_classes, logger, switch_wandb):
        self.device = device
        self.args = args
        self.logger = logger
        self.is_wandb_used = switch_wandb

        self.n_classes = n_classes

        self.net_dataidx_map = net_dataidx_map
        self.train_global = train_global
        self.test_global = test_global
        self.test_sample_number = len(test_global)

        # default: args.init_channels = 16, CIFAR_CLASSES = 10, args.layers = 8
        self.criterion = nn.CrossEntropyLoss().to(device)

        if args.stage == "search":
            self.model_global = Network(args.init_channels, n_classes, args.layers, self.criterion, self.device)
            self.model_global.train()
        else:
            genotype = genotypes.FedNAS_V1
            self.logger.info(genotype)
            self.model_global = NetworkCIFAR(args.init_channels, n_classes, args.layers, args.auxiliary, genotype)
            self.logger.info("param size = %fMB", utils.count_parameters_in_MB(self.model_global))
            # wandb.run.summary["param_size"] = utils.count_parameters_in_MB(self.model_global)

        self.all_train_data_num = sum([len(net_dataidx_map[r]) for r in range(args.client_number)])

        self.client_list = []
        self.setup_clients()

        self.train_acc_avg = 0.0
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0

        self.best_accuracy = 0
        self.best_accuracy_different_cnn_counts = dict()
        self.wandb_table = wandb.Table(columns=["Epoch", "Searched Architecture"])

    def setup_clients(self):
        self.logger.info("############setup_clients (START)#############")
        args_datadir = "./data/cifar10"
        for client_idx in range(self.args.client_number):
            self.logger.info("######client idx = " + str(client_idx))

            dataidxs = self.net_dataidx_map[client_idx]
            local_sample_number = len(dataidxs)

            split = int(np.floor(0.5 * local_sample_number))  # split index
            train_idxs = dataidxs[0:split]
            test_idxs = dataidxs[split:local_sample_number]

            train_local, _ = get_dataloader(self.args.dataset, args_datadir, self.args.batch_size, self.args.batch_size,
                                            train_idxs)
            self.logger.info("client_idx = %d, batch_num_train_local = %d" % (client_idx, len(train_local)))

            test_local, _ = get_dataloader(self.args.dataset, args_datadir, self.args.batch_size, self.args.batch_size,
                                           test_idxs)
            self.logger.info("client_idx = %d, batch_num_test_local = %d" % (client_idx, len(test_local)))

            self.logger.info('n_sample: %d' % local_sample_number)
            self.logger.info('n_training: %d' % len(train_local))
            self.logger.info('n_test: %d' % len(test_local))

            c = Client(client_idx, train_local, test_local, local_sample_number, self.args, self.logger,
                       self.device,
                       self.is_wandb_used)
            self.client_list.append(c)

        self.logger.info("############setup_clients (END)#############")

    def search(self):
        for round_idx in range(self.args.comm_round):
            self.logger.info("Communication round : {}".format(round_idx))

            self.model_global.train()
            w_locals, alpha_locals, acc_locals = [], [], []
            for idx, client in enumerate(self.client_list):
                if self.args.stage == "search":
                    w, alpha, acc = client.local_search(net=copy.deepcopy(self.model_global).to(self.device))
                else:
                    w, acc = client.local_train(net=copy.deepcopy(self.model_global).to(self.device))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                if self.args.stage == "search":
                    alpha_locals.append((client.get_sample_number(), copy.deepcopy(alpha)))
                acc_locals.append(copy.deepcopy(acc))

            if self.args.stage == "search":
                alpha_glob = self.aggregate_alpha(alpha_locals)
                # self.logger.info("global alpha = " + str(alpha_glob))
                # copy alpha to self.model_global
                for a_g, model_arch in zip(alpha_glob, self.model_global.arch_parameters()):
                    model_arch.data.copy_(a_g.data)

            # update global weights
            w_glob = self.aggregate_weight(w_locals)
            # self.logger.info("global weights = " + str(w_glob))

            # copy weight to self.model_global
            self.model_global.load_state_dict(w_glob)

            # local training accuracy
            acc_avg = sum(acc_locals) / len(acc_locals)
            self.logger.info('Round {:3d}, Training Accuracy {:.3f}'.format(round_idx, acc_avg))
            wandb.log({"Train Accuracy": self.test_acc_avg, "Round": round_idx})

            # global test
            self.infer(round_idx)

            if self.args.stage == "search":
                # record the global architecture
                self.record_model_global_architecture(self.model_global, round_idx)

    def aggregate_weight(self, w_locals):
        self.logger.info("################aggregate_weight: %d" % len(w_locals))
        (num0, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / self.all_train_data_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def aggregate_alpha(self, alpha_locals):
        self.logger.info("################aggregate_alpha: %d" % len(alpha_locals))

        (num0, averaged_alphas) = alpha_locals[0]
        for index, alpha in enumerate(averaged_alphas):
            for i in range(0, len(alpha_locals)):
                local_sample_number, local_alpha = alpha_locals[i]
                w = local_sample_number / self.all_train_data_num
                if i == 0:
                    alpha = local_alpha[index] * w
                else:
                    alpha += local_alpha[index] * w
        return averaged_alphas

    # def local_test(self, model_global, round_idx):
    #     train_avg_acc = self.local_test_on_training_data(model_global, round_idx)
    #     valid_avg_acc = self.local_test_on_test_data(model_global, round_idx)
    #     return train_avg_acc, valid_avg_acc
    #
    # def local_test_on_training_data(self, model_global):
    #     num_samples = []
    #     tot_corrects = []
    #     losses = []
    #     for c in self.client_list:
    #         tot_correct, num_sample, loss = c.local_test(model_global, False)
    #
    #         tot_corrects.append(copy.deepcopy(tot_correct))
    #         num_samples.append(copy.deepcopy(num_sample))
    #         losses.append(copy.deepcopy(loss))
    #
    #     train_acc = sum(tot_corrects) / sum(num_samples)
    #     train_loss = sum(losses) / sum(num_samples)
    #
    #     stats = {'training_acc': train_acc, 'training_loss': train_loss, 'num_samples': num_samples}
    #     self.logger.info(stats)
    #     return train_acc
    #
    # def local_test_on_test_data(self, model_global):
    #     num_samples = []
    #     tot_corrects = []
    #     losses = []
    #     for c in self.client_list:
    #         tot_correct, num_sample, loss = c.local_test(model_global, True)
    #
    #         tot_corrects.append(copy.deepcopy(tot_correct))
    #         num_samples.append(copy.deepcopy(num_sample))
    #         losses.append(copy.deepcopy(loss))
    #
    #     train_acc = sum(tot_corrects) / sum(num_samples)
    #     train_loss = sum(losses) / sum(num_samples)
    #
    #     stats = {'test_acc': train_acc, 'test_loss': train_loss, 'num_samples': num_samples}
    #     self.logger.info(stats)
    #     return train_acc

    def infer(self, round_idx):
        self.model_global.eval()
        self.model_global.to(self.device)
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

                    pred = self.model_global(x)
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
                self.logger.info("server test. round_idx = %d, test_loss = %s" % (round_idx, test_loss))

            self.test_acc_avg = test_correct / test_sample_number
            self.test_loss_avg = test_loss
            self.logger.info('searching_valid_acc %f', self.test_acc_avg)
            wandb.log({"Validation Accuracy": self.test_acc_avg, "Round": round_idx})

            end_time = time.time()
            self.logger.info("server_infer time cost: %d" % (end_time - start_time))

    def record_model_global_architecture(self, net, round_idx):
        # save the structure
        genotype, normal_cnn_count, reduce_cnn_count = net.genotype()
        cnn_count = normal_cnn_count + reduce_cnn_count
        wandb.log({"cnn_count": cnn_count, "Round": round_idx})

        self.logger.info("(n:%d,r:%d)" % (normal_cnn_count, reduce_cnn_count))
        self.logger.info('genotype = %s', genotype)
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

    def save_model(self):
        torch.save(self.model_global.state_dict(), "./searched_model.pt")
