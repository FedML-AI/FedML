import logging
import time
import numpy as np
from fedml.core import Context


class FAAggregator(object):
    """
    The FAAggregator class handles the aggregation of local models and sample numbers from clients.

    Args:
        all_train_data_num (int): The total number of training data samples.
        train_data_local_dict (dict): A dictionary containing the local training data for each client.
        train_data_local_num_dict (dict): A dictionary containing the number of local training data samples for each client.
        client_num (int): The number of clients.
        args: Additional arguments.
        server_aggregator: The server aggregator responsible for aggregation.

    Attributes:
        aggregator: The server aggregator responsible for aggregation.
        args: Additional arguments.
        all_train_data_num (int): The total number of training data samples.
        train_data_local_dict (dict): A dictionary containing the local training data for each client.
        train_data_local_num_dict (dict): A dictionary containing the number of local training data samples for each client.
        client_num (int): The number of clients.
        model_dict (dict): A dictionary containing the model parameters from each client.
        sample_num_dict (dict): A dictionary containing the number of samples from each client.
        flag_client_model_uploaded_dict (dict): A dictionary tracking whether each client has uploaded its model.

    Methods:
        get_init_msg(): Get the initialization message from the server aggregator.
        set_init_msg(init_msg): Set the initialization message in the server aggregator.
        get_server_data(): Get the server data from the server aggregator.
        set_server_data(server_data): Set the server data in the server aggregator.
        add_local_trained_result(index, model_params, sample_num): Add local model parameters and sample numbers from a client.
        check_whether_all_receive(): Check if all clients have uploaded their models.
        aggregate(): Aggregate local models and calculate the global result.
        data_silo_selection(round_idx, client_num_in_total, client_num_per_round): Select data silos for clients in a round.
        client_selection(round_idx, client_id_list_in_total, client_num_per_round): Select clients for a round.
        client_sampling(round_idx, client_num_in_total, client_num_per_round): Sample clients for a round.
    """

    def __init__(
        self,
        all_train_data_num,
        train_data_local_dict,
        train_data_local_num_dict,
        client_num,
        args,
        server_aggregator,
    ):
        self.aggregator = server_aggregator
        self.args = args
        self.all_train_data_num = all_train_data_num
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.client_num = client_num
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_init_msg(self):
        """
        Get the initialization message from the server aggregator.

        Returns:
            Any: The initialization message.
        """
        return self.aggregator.get_init_msg()

    def set_init_msg(self, init_msg):
        """
        Set the initialization message in the server aggregator.

        Args:
            init_msg: The initialization message to set.

        Returns:
            None
        """
        self.aggregator.set_init_msg(init_msg)

    def get_server_data(self):
        """
        Get the server data from the server aggregator.

        Returns:
            Any: The server data.
        """
        return self.aggregator.get_server_data()

    def set_server_data(self, server_data):
        """
        Set the server data in the server aggregator.

        Args:
            server_data: The server data to set.

        Returns:
            None
        """
        self.aggregator.set_server_data(server_data)

    def add_local_trained_result(self, index, model_params, sample_num):
        """
        Add local model parameters and sample numbers from a client.

        Args:
            index (int): The index of the client.
            model_params: The local model parameters.
            sample_num (int): The number of samples used for training.

        Returns:
            None
        """
        logging.info("add_model. index = %d" % index)
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
        Aggregate local models and calculate the global result.

        Returns:
            tuple: A tuple containing the global result and a list of local results.
        """
        start_time = time.time()

        local_result_list = []
        for idx in range(self.client_num):
            local_result_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        Context().add(Context.KEY_CLIENT_MODEL_LIST, local_result_list)
        global_result = self.aggregator.aggregate(local_result_list)
        self.set_server_data(global_result)

        end_time = time.time()
        logging.info(f"aggregation result = {global_result}")
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return global_result, local_result_list

    def data_silo_selection(self, round_idx, client_num_in_total, client_num_per_round):
        """
        Select data silos for clients in a round.

        Args:
            round_idx (int): The round index, starting from 0.
            client_num_in_total (int): The total number of clients.
            client_num_per_round (int): The number of clients that can train in a round.

        Returns:
            list: A list of data silo indexes.
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
        Select clients for a round.

        Args:
            round_idx (int): The round index, starting from 0.
            client_id_list_in_total (list): A list of real edge IDs or client indices.
            client_num_per_round (int): The number of clients to select.

        Returns:
            list: A list of selected client IDs.
        """
        if client_num_per_round == len(client_id_list_in_total):
            return client_id_list_in_total
        np.random.seed(round_idx)  # Make sure for each comparison, we are selecting the same clients each round
        client_id_list_in_this_round = np.random.choice(client_id_list_in_total, client_num_per_round, replace=False)
        return client_id_list_in_this_round

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        """
        Sample clients for a round.

        Args:
            round_idx (int): The round index, starting from 0.
            client_num_in_total (int): The total number of clients.
            client_num_per_round (int): The number of clients to sample.

        Returns:
            list: A list of sampled client indices.
        """
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # Make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
    