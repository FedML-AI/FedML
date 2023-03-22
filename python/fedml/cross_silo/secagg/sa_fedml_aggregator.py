import copy
import logging
import random
import time

import numpy as np
import torch
import wandb
from fedml import mlops

from ...core.mpc.secagg import (
    BGW_decoding,
    transform_finite_to_tensor,
    model_dimension,
)


class SecAggAggregator(object):
    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        client_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.client_num = client_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.flag_client_ss_uploaded_dict = dict()

        # for secagg
        self.num_pk_per_user = 2
        self.targeted_number_active_clients = args.worker_num
        self.privacy_guarantee = int(np.floor(args.worker_num / 2))
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter
        self.public_key_others = np.empty(self.num_pk_per_user * self.args.worker_num).astype("int64")
        self.b_u_SS_others = np.empty((self.args.worker_num, self.args.worker_num), dtype="int64")
        self.s_sk_SS_others = np.empty((self.args.worker_num, self.args.worker_num), dtype="int64")

        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
            self.flag_client_ss_uploaded_dict[idx] = False

        self.total_dimension = None
        self.dimensions = []

    def get_global_model_params(self):
        global_model_params = self.trainer.get_model_params()
        self.dimensions, self.total_dimension = model_dimension(global_model_params)
        return global_model_params

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        # for key in model_params.keys():
        #     model_params[key] = model_params[key].to(self.device)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate_mask_reconstruction(self, active_clients, SS_rx, public_key_list):
        """
        Recover the aggregate-mask via decoding
        """
        d = self.total_dimension
        T = self.privacy_guarantee
        p = self.prime_number
        logging.debug("d = {}, T = {}, p = {}".format(d, T, p))

        aggregated_mask = 0

        for i in range(self.targeted_number_active_clients):
            if self.flag_client_model_uploaded_dict[i]:
                SS_input = np.reshape(SS_rx[i, active_clients[: T + 1]], (T + 1, 1))
                b_u = BGW_decoding(SS_input, active_clients[: T + 1], p)
                np.random.seed(b_u[0][0])
                mask = np.random.randint(0, p, size=d).astype(int)
                aggregated_mask += mask
                # z = np.mod(z - temp, p)
            else:
                mask = np.zeros(d, dtype="int")
                SS_input = np.reshape(SS_rx[i, active_clients[: T + 1]], (T + 1, 1))
                s_sk_dec = BGW_decoding(SS_input, active_clients[: T + 1], p)
                for j in range(self.targeted_number_active_clients):
                    s_pk_list_ = public_key_list[1, :]
                    s_uv_dec = np.mod(s_sk_dec[0][0] * s_pk_list_[j], p)
                    # logging.info("&&&&&&&&&&&&&&&&&&&&&&&")
                    # logging.info(s_uv_dec)
                    # logging.info("{},{}".format(i, j))
                    if j == i:
                        temp = np.zeros(d, dtype="int")
                    elif j < i:
                        np.random.seed(s_uv_dec)
                        temp = -np.random.randint(0, p, size=d).astype(int)
                    else:
                        # np.random.seed(s_uv[j-1])
                        np.random.seed(s_uv_dec)
                        temp = +np.random.randint(0, p, size=d).astype(int)
                    # print 'seed, temp=',s_uv_dec,temp
                    mask = np.mod(mask + temp, p)
                # print 'mask =', mask
                aggregated_mask += mask
            aggregated_mask = np.mod(aggregated_mask, p)

        return aggregated_mask

    def aggregate_model_reconstruction(
        self, active_clients_first_round, active_clients_second_round, SS_rx, public_key_list
    ):
        start_time = time.time()
        aggregate_mask = self.aggregate_mask_reconstruction(active_clients_second_round, SS_rx, public_key_list)
        p = self.prime_number
        q_bits = self.precision_parameter
        logging.info("Server starts the reconstruction of aggregate_model")
        # averaged_params = {}
        averaged_params = self.model_dict[active_clients_first_round[0]]
        pos = 0

        for j, k in enumerate(self.model_dict[active_clients_first_round[0]]):
            # averaged_params[k] = 0
            for i, client_idx in enumerate(active_clients_first_round):
                if not (
                    client_idx in self.flag_client_model_uploaded_dict
                    and self.flag_client_model_uploaded_dict[client_idx]
                ):
                    continue
                local_model_params = self.model_dict[client_idx]
                if i == 0:
                    averaged_params[k] = local_model_params[k]
                else:
                    averaged_params[k] += local_model_params[k]
                averaged_params[k] = np.mod(averaged_params[k], p)

            cur_shape = np.shape(averaged_params[k])
            d = self.dimensions[j]
            #aggregate_mask = aggregate_mask.reshape((aggregate_mask.shape[0], 1))
            # logging.info('aggregate_mask shape = {}'.format(np.shape(aggregate_mask)))
            cur_mask = np.array(aggregate_mask[pos : pos + d])
            cur_mask = np.reshape(cur_mask, cur_shape)

            # Cancel out the aggregate-mask to recover the aggregate-model
            averaged_params[k] -= cur_mask
            averaged_params[k] = np.mod(averaged_params[k], p)
            pos += d


        # Convert the model from finite to real
        logging.info("Server converts the aggregate_model from finite to tensor")
        averaged_params = transform_finite_to_tensor(averaged_params, p, q_bits)
        # do the avg after transform
        for j, k in enumerate(averaged_params):
            w = 1 / len(active_clients_first_round)
            averaged_params[k] = averaged_params[k] * w

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def data_silo_selection(self, round_idx, client_num_in_total, client_num_per_round):
        """

        Args:
            round_idx: round index, starting from 0
            client_num_in_total: this is equal to the users in a synthetic data,
                                    e.g., in synthetic_1_1, this value is 30
            client_num_per_round: the number of edge devices that can train

        Returns:
            data_silo_index_list: e.g., when client_num_in_total = 30, client_num_in_total = 3,
                                        this value is the form of [0, 11, 20]

        """
        logging.info(
            "client_num_in_total = %d, client_num_per_round = %d" % (client_num_in_total, client_num_per_round)
        )
        assert client_num_in_total >= client_num_per_round

        if client_num_in_total == client_num_per_round:
            return [i for i in range(client_num_per_round)]
        else:
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            data_silo_index_list = np.random.choice(range(client_num_in_total), client_num_per_round, replace=False)
            return data_silo_index_list

    def client_selection(self, round_idx, client_id_list_in_total, client_num_per_round):
        """
        Args:
            round_idx: round index, starting from 0
            client_id_list_in_total: this is the real edge IDs.
                                    In MLOps, its element is real edge ID, e.g., [64, 65, 66, 67];
                                    in simulated mode, its element is client index starting from 1, e.g., [1, 2, 3, 4]
            client_num_per_round:

        Returns:
            client_id_list_in_this_round: sampled real edge ID list, e.g., [64, 66]
        """
        if client_num_per_round == len(client_id_list_in_total):
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

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        # if self.trainer.test_on_the_server(
        #     self.train_data_local_dict,
        #     self.test_data_local_dict,
        #     self.device,
        #     self.args,
        # ):
        #     return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = (
                    metrics["test_correct"],
                    metrics["test_total"],
                    metrics["test_loss"],
                )
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            if self.args.enable_wandb:
                wandb.log({"Train/Acc": train_acc, "round": round_idx})
                wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {"training_acc": train_acc, "training_loss": train_loss}
            logging.info(stats)

            mlops.log({"accuracy": round(train_acc, 4), "loss": round(train_loss, 4)})

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)

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
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            logging.info(stats)
