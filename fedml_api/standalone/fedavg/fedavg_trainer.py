import copy

import torch
import torchvision.models as models
import wandb
from torch import nn

from fedml_api.standalone.fedavg.client import Client
from fedml_api.standalone.fedavg.data_loader import get_dataloader


class FedAvgTrainer(object):
    def __init__(self, net_dataidx_map, train_global, test_global, device, args, n_classes, logger, switch_wandb):
        self.device = device
        self.args = args
        self.logger = logger
        self.switch_wandb = switch_wandb

        self.n_classes = n_classes

        self.net_dataidx_map = net_dataidx_map
        self.train_global = train_global
        self.test_global = test_global
        self.test_sample_number = len(test_global)

        self.all_train_data_num = sum([len(net_dataidx_map[r]) for r in range(args.client_number)])

        if args.model == "resnet":
            self.model_global = models.resnet152(pretrained=False)
        elif args.model == "vgg":
            self.model_global = models.vgg11(pretrained=False)
        elif args.model == "densenet":
            self.model_global = models.densenet201(pretrained=False)  # 1/3 parameter number of ResNet
        self.model_global.train()

        self.client_list = []
        self.setup_clients()

    def setup_clients(self):
        self.logger.info("############setup_clients (START)#############")
        args_datadir = "./data/cifar10"
        for client_idx in range(self.args.client_number):
            self.logger.info("######client idx = " + str(client_idx))
            dataidxs = self.net_dataidx_map[client_idx]
            local_sample_number = len(dataidxs)

            # training batch size = 64; test batch size = 32
            train_dl_local, test_dl_local = get_dataloader(self.args.dataset, args_datadir, self.args.batch_size, 32,
                                                           dataidxs)
            self.logger.info('n_sample: %d' % local_sample_number)
            self.logger.info('n_training: %d' % len(train_dl_local))
            self.logger.info('n_test: %d' % len(test_dl_local))

            c = Client(train_dl_local, test_dl_local, local_sample_number, self.args,
                       self.logger, self.device)
            self.client_list.append(c)

        self.logger.info("############setup_clients (END)#############")

    def train(self):
        for round_idx in range(self.args.comm_round):
            self.logger.info("Communication round : {}".format(round_idx))

            self.model_global.train()
            w_locals, loss_locals = [], []
            for idx, client in enumerate(self.client_list):
                w, loss = client.train(net=copy.deepcopy(self.model_global).to(self.device))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                loss_locals.append(copy.deepcopy(loss))

            # update global weights
            w_glob = self.aggregate(w_locals)
            # self.logger.info("global weights = " + str(w_glob))

            # copy weight to net_glob
            self.model_global.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            self.logger.info('Round {:3d}, Average loss {:.3f}'.format(round_idx, loss_avg))

            self.local_test(self.model_global, round_idx)

    def aggregate(self, w_locals):
        self.logger.info("################aggregate: %d" % len(w_locals))
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

        wandb.log({"Train Accuracy": train_acc, "Round": round_idx})
        wandb.log({"Train Loss": train_loss, "Round": round_idx})

        stats = {'training_acc': train_acc, 'training_loss': train_loss, 'num_samples': num_samples}
        self.logger.info(stats)

    def local_test_on_test_data(self, model_global, round_idx):
        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.client_list:
            tot_correct, num_sample, loss = c.local_test(model_global, True)

            tot_corrects.append(copy.deepcopy(tot_correct))
            num_samples.append(copy.deepcopy(num_sample))
            losses.append(copy.deepcopy(loss))

        train_acc = sum(tot_corrects) / sum(num_samples)
        train_loss = sum(losses) / sum(num_samples)

        wandb.log({"Validation Accuracy": train_acc, "Round": round_idx})
        wandb.log({"Validation Loss": train_loss, "Round": round_idx})

        stats = {'test_acc': train_acc, 'test_loss': train_loss, 'num_samples': num_samples}
        self.logger.info(stats)

    def global_test(self):
        self.logger.info("################global_test")
        acc_train, num_sample, loss_train = self.test_using_global_dataset(self.model_global, self.train_global,
                                                                           self.device)
        acc_train = acc_train / num_sample

        acc_test, num_sample, loss_test = self.test_using_global_dataset(self.model_global, self.test_global,
                                                                         self.device)
        acc_test = acc_test / num_sample

        self.logger.info("Global Training Accuracy: {:.2f}".format(acc_train))
        self.logger.info("Global Testing Accuracy: {:.2f}".format(acc_test))
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
