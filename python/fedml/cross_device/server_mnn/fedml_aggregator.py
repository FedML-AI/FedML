import logging
from .utils import read_mnn_as_tensor_dict
import copy
import time

import MNN
import numpy as np
import torch
import wandb

from fedml import mlops

F = MNN.expr
nn = MNN.nn


class FedMLAggregator(object):
    def __init__(
        self, test_dataloader, worker_num, device, args, aggregator,
    ):
        self.aggregator = aggregator

        self.args = args
        self.test_global = test_dataloader

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.aggregator.get_model_params()

    # TODO: refactor MNN-related file processing
    def get_global_model_params_file(self):
        return self.aggregator.get_model_params_file()

    def set_global_model_params(self, model_parameters):
        logging.info("FedDebug. model_parameters = {}".format(model_parameters))
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.info("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def _test_individual_model_perf_before_agg(self, model_file_path, round_idx):
        self.test_on_server_for_all_clients_mnn(model_file_path, round_idx, report_metrics=False)

    def aggregate(self):
        logging.info("FedMLDebug. Individual model performance:")
        for idx in range(self.worker_num):
            logging.info("self.model_dict[idx] = {}".format(self.model_dict[idx]))
            mnn_file_path = self.model_dict[idx]
            self._test_individual_model_perf_before_agg(mnn_file_path, self.args.round_idx)

        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            logging.info("self.model_dict[idx] = {}".format(self.model_dict[idx]))
            mnn_file_path = self.model_dict[idx]
            tensor_params_dict = read_mnn_as_tensor_dict(mnn_file_path)
            model_list.append((self.sample_num_dict[idx], tensor_params_dict))
            training_num += self.sample_num_dict[idx]
        logging.info("training_num = {}".format(training_num))
        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        averaged_params = self.aggregator.aggregate(model_list)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def data_silo_selection(self, round_idx, data_silo_num_in_total, client_num_in_total):
        """

        Args:
            round_idx: round index, starting from 0
            data_silo_num_in_total: this is equal to the users in a synthetic data,
                                    e.g., in synthetic_1_1, this value is 30
            client_num_in_total: the number of edge devices that can train

        Returns:
            data_silo_index_list: e.g., when data_silo_num_in_total = 30, client_num_in_total = 3,
                                        this value is the form of [0, 11, 20]

        """
        logging.info(
            "data_silo_num_in_total = %d, client_num_in_total = %d" % (data_silo_num_in_total, client_num_in_total)
        )
        assert data_silo_num_in_total >= client_num_in_total
        if client_num_in_total == data_silo_num_in_total:
            return [i for i in range(data_silo_num_in_total)]
        else:
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            data_silo_index_list = np.random.choice(range(data_silo_num_in_total), client_num_in_total, replace=False)
        return data_silo_index_list

    def client_selection(self, round_idx, client_id_list_in_total, client_num_per_round):
        """
        Args:
            round_idx: round index, starting from 0
            client_id_list_in_total: this is the real edge IDs.
                                    In MLOps, its element is real edge ID, e.g., [64, 65, 66, 67];
                                    in simulated mode, its element is client index starting from 0, e.g., [0, 1, 2, 3]
            client_num_per_round:

        Returns:
            client_id_list_in_this_round: sampled real edge ID list, e.g., [64, 66]
        """
        if client_num_per_round == len(client_id_list_in_total) or len(client_id_list_in_total) == 1:  # for debugging
            return client_id_list_in_total
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_id_list_in_this_round = np.random.choice(client_id_list_in_total, client_num_per_round, replace=False)
        return client_id_list_in_this_round

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics

    def test(self, test_data, device, args):
        # test data
        test_num_samples = []
        test_tot_corrects = []
        test_losses = []

        metrics = self._test(test_data, device, args)

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
            wandb.log({"Test/Acc": test_acc, "round": args.round_idx})
            wandb.log({"Test/Loss": test_loss, "round": args.round_idx})

        mlops.log({"Test/Acc": test_acc, "round": args.round_idx})
        mlops.log({"Test/Loss": test_loss, "round": args.round_idx})

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)

        return (test_acc, test_loss, None, None)

    def test_on_server_for_all_clients(self, round_idx, global_model_file=None):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            self.aggregator.test_all(
                self.train_data_local_dict,
                self.test_data_local_dict,
                self.device,
                self.args,
            )

            if round_idx == self.args.comm_round - 1:
                # we allow to return four metrics, such as accuracy, AUC, loss, etc.
                metric_result_in_current_round = self.aggregator.test(self.test_global, self.device, self.args)
            else:
                metric_result_in_current_round = self.aggregator.test(self.val_global, self.device, self.args)
            logging.info("metric_result_in_current_round = {}".format(metric_result_in_current_round))

            if round_idx == self.args.comm_round - 1:
                mlops.log({"round_idx": round_idx})
        else:
            mlops.log({"round_idx": round_idx})

    def test_on_server_for_all_clients_mnn(self, mnn_file_path, round_idx, report_metrics=True):
        # load global model from MNN
        var_map = F.load_as_dict(mnn_file_path)
        input_dicts, output_dicts = F.get_inputs_and_outputs(var_map)
        input_names = [n for n in input_dicts.keys()]
        output_names = [n for n in output_dicts.keys()]
        input_vars = [input_dicts[n] for n in input_names]
        output_vars = [output_dicts[n] for n in output_names]
        module = MNN.nn.load_module(input_vars, output_vars, False)

        module.train(False)
        self.test_global.reset()

        correct = 0
        for i in range(self.test_global.iter_number):
            example = self.test_global.next()
            input_data = example[0]
            output_target = example[1]
            data = input_data[0]  # which input, model may have more than one inputs
            label = output_target[0]  # also, model may have more than one outputs

            result = module.forward(data)
            predict = F.argmax(result, 1)
            predict = np.array(predict.read())

            label_test = np.array(label.read())
            correct += np.sum(label_test == predict)

            target = F.one_hot(F.cast(label, F.int), 10, 1, 0)
            loss = nn.loss.cross_entropy(result, target)

        logging.info(f"correct = {correct}, self.test_global.size = {self.test_global.size}")
        test_accuracy = correct / self.test_global.size
        test_loss = loss.read()

        if report_metrics:
            logging.info("test acc = {}".format(test_accuracy))
            logging.info("test loss = {}, round loss {}".format(test_loss, round(float(np.round(test_loss, 4)), 4)))

            mlops.log(
                {
                    "round_idx": round_idx,
                    "accuracy": round(float(np.round(test_accuracy, 4)), 4),
                    "loss": round(float(np.round(test_loss, 4)), 4),
                }
            )

            if self.args.enable_wandb:
                wandb.log(
                    {"round idx": round_idx, "test acc": test_accuracy, "test loss": test_loss, }
                )
