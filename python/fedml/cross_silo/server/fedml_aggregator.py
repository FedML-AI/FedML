import logging
import random
import time

import numpy as np
import torch

from fedml import mlops
from ...core import Context
from ...ml.engine import ml_engine_adapter


class FedMLAggregator(object):
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
        server_aggregator,
    ):
        self.aggregator = server_aggregator

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        Context().add(Context.KEY_TEST_DATA, self.val_global)

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.client_num = client_num
        self.device = device
        self.args.device = device
        logging.info("self.device = {}".format(self.device))
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.is_fhe_enabled = hasattr(args, "enable_fhe") and args.enable_fhe

    def get_global_model_params(self):
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)

        # for dictionary model_params, we let the user level code to control the device
        if type(model_params) is not dict and (not self.is_fhe_enabled):
            model_params = ml_engine_adapter.model_params_to_device(self.args, model_params, self.device)

        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug("client_num = {}".format(self.client_num))
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
        # model_list is the list after outlier removal
        model_list, model_list_idxes = self.aggregator.on_before_aggregation(model_list)
        Context().add(Context.KEY_CLIENT_MODEL_LIST, model_list)

        averaged_params = self.aggregator.aggregate(model_list)

        if type(averaged_params) is dict:
            if len(averaged_params) == self.client_num + 1: # aggregator pass extra {-1 : global_parms_dict}  as global_params
                itr_count = len(averaged_params) - 1        # do not apply on_after_aggregation to client -1
            else:
                itr_count = len(averaged_params)

            for client_index in range(itr_count):
                averaged_params[client_index] = self.aggregator.on_after_aggregation(averaged_params[client_index])
        else:
            averaged_params = self.aggregator.on_after_aggregation(averaged_params)

        if not self.is_fhe_enabled:
            self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params, model_list, model_list_idxes

    def assess_contribution(self):
        if hasattr(self.args, "enable_contribution") and \
                self.args.enable_contribution is not None and self.args.enable_contribution:
            self.aggregator.assess_contribution()

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
        if self.is_fhe_enabled:
            logging.info("Encrypted global model cannot be tested on the server")
            return

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
            metric_results_in_the_last_round = Context().get(Context.KEY_METRICS_ON_AGGREGATED_MODEL)
            Context().add(Context.KEY_METRICS_ON_AGGREGATED_MODEL, metric_result_in_current_round)
            if metric_results_in_the_last_round is not None:
                Context().add(Context.KEY_METRICS_ON_LAST_ROUND, metric_results_in_the_last_round)
            else:
                Context().add(Context.KEY_METRICS_ON_LAST_ROUND, metric_result_in_current_round)
            key_metrics_on_last_round = Context().get(Context.KEY_METRICS_ON_LAST_ROUND)
            logging.info("key_metrics_on_last_round = {}".format(key_metrics_on_last_round))

            if round_idx == self.args.comm_round - 1:
                mlops.log({"round_idx": round_idx})
        else:
            mlops.log({"round_idx": round_idx})
    
    def get_dummy_input_tensor(self):
        test_data = None
        if self.test_global:
            test_data = self.test_global
        else:   # if test_global is None, then we use the first non-empty test_data_local_dict
            for k, v in self.test_data_local_dict.items():
                if v:
                    test_data = v
                    break 
        
        with torch.no_grad():
            batch_idx, features_label_tensors = next(enumerate(test_data))  # test_data -> dataloader obj
            dummy_list = []
            for tensor in features_label_tensors:
                dummy_tensor = tensor[:1] # only take the first element as dummy input
                dummy_list.append(dummy_tensor)
        features = dummy_list[:-1]  # Can adapt Process Multi-Label
        return features

    def get_input_shape_type(self):
        test_data = None
        if self.test_global:
            test_data = self.test_global
        else:   # if test_global is None, then we use the first non-empty test_data_local_dict
            for k, v in self.test_data_local_dict.items():
                if v:
                    test_data = v
                    break
        
        with torch.no_grad():
            batch_idx, features_label_tensors = next(enumerate(test_data))  # test_data -> dataloader obj
            dummy_list = []
            for tensor in features_label_tensors:
                dummy_tensor = tensor[:1] # only take the first element as dummy input
                dummy_list.append(dummy_tensor)
        features = dummy_list[:-1]  # Can adapt Multi-Label

        input_shape, input_type = [], []
        for feature in features:
            input_shape.append(list(feature.shape))
            if feature.dtype == torch.int or feature.dtype == torch.int8 or feature.dtype == torch.int16 or \
                    feature.dtype == torch.int32 or feature.dtype == torch.int64 or feature.dtype == torch.uint8 or \
                    feature.dtype == torch.short or feature.dtype == torch.long or feature.dtype == torch.bool:
                input_type.append("int")
            else:
                input_type.append("float")
            
        return input_shape, input_type
    
    def save_dummy_input_tensor(self):
        import pickle
        features = self.get_input_size_type()
        with open('dummy_input_tensor.pkl', 'wb') as handle:
            pickle.dump(features, handle)

        # TODO: save the dummy_input_tensor.pkl to s3, and transfer when click "Create Model Card"
