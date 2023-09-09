import logging
import random
import time

import numpy as np
import torch

from fedml import mlops
from ...core import Context
from ...ml.engine import ml_engine_adapter


class FedMLAggregator(object):
    """
    A class for federated machine learning aggregation and related tasks.

    Args:
        train_global: Global training data.
        test_global: Global testing data.
        all_train_data_num: Number of samples in the entire training dataset.
        train_data_local_dict: Local training data dictionary.
        test_data_local_dict: Local testing data dictionary.
        train_data_local_num_dict: Number of local samples for each client.
        client_num: Number of clients.
        device: Device to run computations (e.g., 'cuda' or 'cpu').
        args: Additional configuration arguments.
        server_aggregator: Aggregator for server-side operations.
    """

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

    def get_global_model_params(self):
        """
        Get the global model parameters.

        Returns:
            dict: Global model parameters.
        """
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        """
        Set the global model parameters.

        Args:
            model_parameters (dict): Global model parameters.
        """
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        """
        Add locally trained model parameters for aggregation.

        Args:
            index (int): Index of the client.
            model_params (dict): Local model parameters.
            sample_num (int): Number of local samples used for training.
        """
        logging.info("add_model. index = %d" % index)

        # For dictionary model_params, let the user-level code control the device
        if type(model_params) is not dict:
            model_params = ml_engine_adapter.model_params_to_device(self.args, model_params, self.device)

        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        """
        Check if all clients have uploaded their models.

        Returns:
            bool: True if all clients have uploaded their models, False otherwise.
        """
        logging.debug("client_num = {}".format(self.client_num))
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        """
        Aggregate local models from clients to obtain a global model.

        Returns:
            tuple: A tuple containing:
                - dict: Averaged global model parameters.
                - list: List of model tuples before aggregation.
                - list: List of indices corresponding to selected models for aggregation.
        """
        start_time = time.time()

        model_list = []
        for idx in range(self.client_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        # Model list is the list after outlier removal
        model_list, model_list_idxes = self.aggregator.on_before_aggregation(model_list)
        Context().add(Context.KEY_CLIENT_MODEL_LIST, model_list)

        averaged_params = self.aggregator.aggregate(model_list)

        if type(averaged_params) is dict:
            if len(averaged_params) == self.client_num + 1:  # Aggregator passes extra {-1: global_parms_dict} as global_params
                itr_count = len(averaged_params) - 1  # Do not apply on_after_aggregation to client -1
            else:
                itr_count = len(averaged_params)

            for client_index in range(itr_count):
                averaged_params[client_index] = self.aggregator.on_after_aggregation(averaged_params[client_index])
        else:
            averaged_params = self.aggregator.on_after_aggregation(averaged_params)

        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params, model_list, model_list_idxes

    def assess_contribution(self):
        """
        Assess the contribution of clients to the global model.
        """
        if hasattr(self.args, "enable_contribution") and \
                self.args.enable_contribution is not None and self.args.enable_contribution:
            self.aggregator.assess_contribution()

    def data_silo_selection(self, round_idx, client_num_in_total, client_num_per_round):
        """
        Select a subset of data silos (clients) for a federated learning round.

        Args:
            round_idx (int): Round index, starting from 0.
            client_num_in_total (int): Total number of clients.
            client_num_per_round (int): Number of clients to select for the current round.

        Returns:
            list: List of selected data silo indices.
        """
        logging.info(
            "client_num_in_total = %d, client_num_per_round = %d" % (client_num_in_total, client_num_per_round)
        )
        assert client_num_in_total >= client_num_per_round

        if client_num_in_total == client_num_per_round:
            return [i for i in range(client_num_per_round)]
        else:
            np.random.seed(round_idx)  # Make sure for each comparison, we are selecting the same clients each round
            data_silo_index_list = np.random.choice(range(client_num_in_total), client_num_per_round, replace=False)
            return data_silo_index_list

    def client_selection(self, round_idx, client_id_list_in_total, client_num_per_round):
        """
        Select a subset of clients for a federated learning round.

        Args:
            round_idx (int): Round index, starting from 0.
            client_id_list_in_total (list): List of real edge IDs or client indices.
            client_num_per_round (int): Number of clients to select for the current round.

        Returns:
            list: List of selected client IDs or indices.
        """
        if client_num_per_round == len(client_id_list_in_total):
            return client_id_list_in_total
        np.random.seed(round_idx)  # Make sure for each comparison, we are selecting the same clients each round
        client_id_list_in_this_round = np.random.choice(client_id_list_in_total, client_num_per_round, replace=False)
        return client_id_list_in_this_round

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        """
        Sample a subset of clients for a federated learning round.

        Args:
            round_idx (int): Round index, starting from 0.
            client_num_in_total (int): Total number of clients.
            client_num_per_round (int): Number of clients to sample for the current round.

        Returns:
            list: List of sampled client indices.
        """
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # Make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        """
        Generate a validation set for model evaluation.

        Args:
            num_samples (int, optional): Number of samples to include in the validation set. Defaults to 10000.

        Returns:
            DataLoader: DataLoader containing the validation set.
        """
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        """
        Perform model testing on the server for all clients in the current round.

        Args:
            round_idx (int): Round index.
        """
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            self.aggregator.test_all(
                self.train_data_local_dict,
                self.test_data_local_dict,
                self.device,
                self.args,
            )

            if round_idx == self.args.comm_round - 1:
                # Allow returning multiple metrics (e.g., accuracy, AUC, loss, etc.) in the final round
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
        else:
            mlops.log({"round_idx": round_idx})
    
    def get_dummy_input_tensor(self):
        """
        Get a dummy input tensor from the test data.

        Returns:
            list: List of dummy input tensors.
        """
        test_data = None
        if self.test_global:
            test_data = self.test_global
        else:  # If test_global is None, use the first non-empty test_data_local_dict
            for k, v in self.test_data_local_dict.items():
                if v:
                    test_data = v
                    break 
        
        with torch.no_grad():
            batch_idx, features_label_tensors = next(enumerate(test_data))  # test_data -> DataLoader object
            dummy_list = []
            for tensor in features_label_tensors:
                dummy_tensor = tensor[:1] # Only take the first element as dummy input
                dummy_list.append(dummy_tensor)
        features = dummy_list[:-1]  # Can adapt to process multi-label data
        return features

    def get_input_shape_type(self):
        """
        Get the shapes and types of input features in the test data.

        Returns:
            tuple: A tuple containing:
                - list: List of input feature shapes.
                - list: List of input feature types ('int' or 'float').
        """
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
        """
        Save the dummy input tensor information to a file.

        This function saves the input shape and type information to a file named 'dummy_input_tensor.pkl'.
        The saved file can be used for reference or documentation purposes.

        Note: To save the file to a specific location (e.g., S3), additional implementation is required.

        Example:
            To save to a specific location (e.g., S3), you can modify this function to upload the file accordingly.

        """
        import pickle
        features = self.get_input_shape_type()
        with open('dummy_input_tensor.pkl', 'wb') as handle:
            pickle.dump(features, handle)

        # TODO: Save the 'dummy_input_tensor.pkl' to S3 or another desired location, and transfer it when needed.
        