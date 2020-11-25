import copy
import logging
import time
import torch
import wandb
import numpy as np
from torch import nn

from fedml_api.distributed.fedseg.utils import transform_list_to_tensor, SegmentationLosses, Evaluator, Saver


class FedSegAggregator(object):
    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device, model, n_class, args):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.evaluator = Evaluator(n_class)
        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.model, _ = self.init_model(model)
        logging.info('Initializing FedSegAggregator with workers: {0}, num_classes:{1}'.format(worker_num, n_class))


    def init_model(self, model):
        model_params = model.state_dict()
        return model, model_params

    def get_global_model_params(self):
        return self.model.state_dict()

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

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

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_all_clients(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            train_acc_clients = []
            train_acc_class_clients = []
            train_mIoU_clients = []
            train_FWIoU_clients = []
            train_num_samples_clients = []
            train_losses_clients = []

            test_acc_clients = []
            test_acc_class_clients = []
            test_mIoU_clients = []
            test_FWIoU_clients = []
            test_num_samples_clients = []
            test_losses_clients = []
            for client_idx in range(self.args.client_num_in_total):

                # train data
                acc, acc_class, mIoU, FWIoU, num_samples, loss = self._infer(self.train_data_local_dict[client_idx])
                train_acc_clients.append(acc)
                train_acc_class_clients.append(acc_class)
                train_mIoU_clients.append(mIoU)
                train_FWIoU_clients.append(FWIoU)
                train_num_samples_clients.append(num_samples)
                train_losses_clients.append(loss)

                # test data
                acc, acc_class, mIoU, FWIoU, num_samples, loss = self._infer(self.test_data_local_dict[client_idx])
                test_acc_clients.append(acc)
                test_acc_class_clients.append(acc_class)
                test_mIoU_clients.append(mIoU)
                test_FWIoU_clients.append(FWIoU)
                test_num_samples_clients.append(num_samples)
                test_losses_clients.append(loss)

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_acc_clients) / len(train_acc_clients)
            train_acc_class = sum(train_acc_class_clients) / len(train_acc_class_clients)
            train_mIoU = sum(train_mIoU_clients) / len(train_mIoU_clients)
            train_FWIoU = sum(train_FWIoU_clients) / len(train_FWIoU_clients)
            train_loss = sum(train_losses_clients) / sum(train_num_samples_clients)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Acc_class": train_acc_class, "round": round_idx})
            wandb.log({"Train/mIoU": train_mIoU, "round": round_idx})
            wandb.log({"Train/FWIoU": train_FWIoU, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 
                     'training_acc_class': train_acc_class,
                     'training_mIoU': train_mIoU,
                     'training_FWIoU': train_FWIoU,  
                     'training_loss': train_loss}
            logging.info(stats)

            # test on test dataset
            test_acc = sum(test_acc_clients) / len(test_acc_clients)
            test_acc_class = sum(test_acc_class_clients) / len(test_acc_class_clients)
            test_mIoU = sum(test_mIoU_clients) / len(test_mIoU_clients)
            test_FWIoU = sum(test_FWIoU_clients) / len(test_FWIoU_clients)
            test_loss = sum(test_losses_clients) / sum(test_num_samples_clients)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Acc_class": test_acc_class, "round": round_idx})
            wandb.log({"Test/mIoU": test_mIoU, "round": round_idx})
            wandb.log({"Test/FWIoU": test_FWIoU, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'testing_acc': test_acc, 
                     'testing_acc_class': test_acc_class,
                     'testing_mIoU': test_mIoU,
                     'testing_FWIoU': test_FWIoU,  
                     'testing_loss': test_loss}
            logging.info(stats)

    def _infer(self, test_data):
        self.model.eval()
        self.model.to(self.device)
        self.evaluator.reset()

        test_loss = test_acc = test_total = 0.
        criterion = SegmentationLosses().build_loss(mode=self.args.loss_type)
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_data):
                x, target = batch['image'], batch['label']
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                loss = criterion(output, target).to(self.device)
                test_loss += loss.item()
                test_total += target.size(0)
                pred = output.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                self.evaluator.add_batch(target, pred)

        
        # Evaluation Metrics
        test_acc = self.evaluator.Pixel_Accuracy()
        test_acc_class = self.evaluator.Pixel_Accuracy_Class()
        test_mIoU = self.evaluator.Mean_Intersection_over_Union()
        test_FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        return test_acc, test_acc_class, test_mIoU, test_FWIoU, test_total, test_loss
