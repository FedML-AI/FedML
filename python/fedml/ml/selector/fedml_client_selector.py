import copy
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
        self.timeout = 60 # 20 * 5 seconds = 5 minutes timeout
        
        self.client_online_status_mapping = client_online_status_mapping
        self.client_real_ids = client_real_ids
        
    def run(self):
        selected_client_in_this_round = []
        selected_data_silo_index_list = []
                
        to_be_selected_client_real_id_list = copy.deepcopy(self.client_real_ids)
        to_be_selected_client_num = self.client_num_per_round
        
        while self.timeout > 0:
            """
            select and pull client status
            """
            print(f"to_be_selected_client_real_id_list = {to_be_selected_client_real_id_list}")
            random_seed = self.round_idx + self.timeout # avoid repeat the same selected devices
            selected_client_list = self.client_selection(
                random_seed, to_be_selected_client_real_id_list, to_be_selected_client_num
            )
            client_idx = 0
            for client_id in selected_client_list:
                self.callback_on_check_client_status(
                    int(client_id), self.client_real_ids.index(int(client_id)),
                )
                client_idx += 1

            """
            wait for 5 seconds
            """
            times_wait_for_online_clients = 5
            is_selected_client_all_onlne = False
            while not is_selected_client_all_onlne and times_wait_for_online_clients > 0:
                is_selected_client_all_onlne = True
                for client_id in selected_client_list:
                    if not self.client_online_status_mapping.get(str(client_id), False):
                        is_selected_client_all_onlne = False
                        break

                logging.info(
                    f"need to select {selected_client_list} clients. Current online clients = {self.client_online_status_mapping}"
                )
                time.sleep(1)
                times_wait_for_online_clients -= 1

            for client_id in selected_client_list:
                client_online_status = self.client_online_status_mapping.get(str(client_id), False)
                if client_online_status:
                    # add online clients to selected_client_in_this_round
                    selected_client_in_this_round.append(int(client_id))
                    
                    # only remove online clients
                    to_be_selected_client_real_id_list.remove(int(client_id))
            
            logging.info(f"selected_client_in_this_round = {selected_client_in_this_round}")
            if len(selected_client_in_this_round) == self.client_num_per_round:
                # find the connected clients and notify the message queue thread
                selected_data_silo_index_list = self.data_silo_selection(
                    self.round_idx, self.client_num_in_total, len(selected_client_in_this_round),
                )
                self.callback_on_success(selected_client_in_this_round, selected_data_silo_index_list)
                break
            else:              
                # update to_be_selected_client_real_id_list and try again
                to_be_selected_client_num = self.client_num_per_round - len(selected_client_in_this_round)  
                # edge case: if there aren't enough client left, we will try again but use the full client real id list
                if len(to_be_selected_client_real_id_list) < to_be_selected_client_num:
                    logging.info("not enough client left, we still try again but use the full client real id list")
                    to_be_selected_client_real_id_list = copy.deepcopy(self.client_real_ids)
                    to_be_selected_client_num = self.client_num_per_round
                    selected_client_in_this_round.clear()
                    
            self.timeout -= 1
        
        # still cannot find enough connected clients, notify exception
        if len(selected_client_in_this_round) != self.client_num_per_round:
            self.callback_on_exception()
        print("thread ended successfully!")
    
    def client_selection(
        self, random_seed, client_id_list_in_total, client_num_per_round
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
        np.random.seed(random_seed)
        client_id_list_in_this_round = np.random.choice(client_id_list_in_total, client_num_per_round, replace=False)
        return client_id_list_in_this_round

    def data_silo_selection(self, random_seed, client_num_in_total, client_num_per_round):
        """

        Args:
            random_seed: random seed
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
            # make sure for each comparison, we are selecting the same clients each round
            np.random.seed(random_seed)  
            data_silo_index_list = np.random.choice(range(client_num_in_total), client_num_per_round, replace=False)
            return data_silo_index_list