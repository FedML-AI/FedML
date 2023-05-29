import logging
import time

import numpy as np
from fedml import mlops

import time
import threading


class FedMLClientSelector(threading.Thread):
    """select client_num_per_round users: 
    1. uniformly random picks M clients which are ONLINE, and send check messages
    2. wait for 5 seconds, if they are all ONLINE, go to send the models
    3. if M' doesn't respond, we continue to select M' clients and wait for their response
    4. if the 2nd try still cannot match with M ONLINE clients, we then allow dropout 10% dropout rate during selection
    
    reference implementation of the threading with callback: https://gist.github.com/amirasaran/e91c7253c03518b8f7b7955df0e954bb
    """

    def __init__(self, round_idx, client_num_in_total, client_num_per_round, client_real_ids, client_online_status_mapping, 
                 callback_on_success=None, callback_on_check_client_status=None, callback_on_exception=None, *args, **kwargs):
        super(FedMLClientSelector, self).__init__(target=self.run, *args, **kwargs)
        self.callback_on_success = callback_on_success
        self.callback_on_check_client_status = callback_on_check_client_status
        self.callback_on_exception = callback_on_exception
        
        self.round_idx = round_idx
        self.client_num_in_total = client_num_in_total
        self.client_num_per_round = client_num_per_round
        self.dropout_rate = 0.1
        self.timeout = 120 # 2 minutes timeout
        
        self.client_online_status_mapping = client_online_status_mapping
        self.client_real_ids = client_real_ids
        
    def run(self):
        # while True:
            # select and check
            
            # wait for 5 seconds
            
            # if all_client_is_online = False
        selected_client_in_this_round = self.client_selection(
            self.round_idx, self.client_real_ids, self.client_num_per_round
        )
        data_silo_index_list = self.data_silo_selection(
            self.round_idx, self.client_num_in_total, len(selected_client_in_this_round),
        )
        client_idx_in_this_round = 0
        for client_id in selected_client_in_this_round:
            try:
                # call back to message thread
                self.callback_on_check_client_status(
                    client_id, data_silo_index_list[client_idx_in_this_round],
                )
                logging.info("Connection ready for client" + str(client_id))
            except Exception as e:
                logging.info("Connection not ready for client" + str(client_id))
            client_idx_in_this_round += 1
            
        print("thread start successfully and sleep for 5 seconds")
        all_client_is_online = False
        while not all_client_is_online:
            logging.info("self.client_online_status_mapping = {}".format(self.client_online_status_mapping))
            all_client_is_online = True
            
            for client_id in selected_client_in_this_round:
                if not self.client_online_status_mapping.get(str(client_id), False):
                    all_client_is_online = False
                    break

            logging.info(
                f"online clients = {self.client_online_status_mapping}, all_client_is_online = {str(all_client_is_online)}"
            )
            time.sleep(1)
        # if all_client_is_online:
        self.callback_on_success(selected_client_in_this_round, data_silo_index_list)
        print("thread ended successfully!")
    
    def client_selection(
        self, round_idx, client_id_list_in_total, client_num_per_round
    ):
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
        # make sure for each comparison, we are selecting the same clients each round
        np.random.seed(round_idx)
        client_id_list_in_this_round = np.random.choice(
            client_id_list_in_total, client_num_per_round, replace=False
        )
        return client_id_list_in_this_round

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
            "client_num_in_total = %d, client_num_per_round = %d"
            % (client_num_in_total, client_num_per_round)
        )
        assert client_num_in_total >= client_num_per_round

        if client_num_in_total == client_num_per_round:
            return [i for i in range(client_num_per_round)]
        else:
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            data_silo_index_list = np.random.choice(
                range(client_num_in_total), client_num_per_round, replace=False
            )
            return data_silo_index_list
        
    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
