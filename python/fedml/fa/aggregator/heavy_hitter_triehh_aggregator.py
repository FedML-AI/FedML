import logging
import math
import numpy as np
from typing import List, Tuple, Any
from fedml.fa.base_frame.server_aggregator import FAServerAggregator

"""
Federated Heavy Hitters Discovery with Differential Privacy: https://arxiv.org/pdf/1902.08534.pdf
reference: https://github.com/google-research/federated/tree/master/triehh
"""


class HeavyHitterTriehhAggregatorFA(FAServerAggregator):
    def __init__(self, args, train_data_num):
        super().__init__(args)
        if hasattr(args, "max_word_len"):
            self.MAX_L = args.max_word_len
        else:
            self.MAX_L = 10
        if hasattr(args, "epsilon"):
            self.epsilon = args.epsilon
        else:
            self.epsilon = 1.0
        if hasattr(args, "delta"):
            self.delta = args.delta
        else:
            self.delta = 2.3e-12
        self.num_runs = args.comm_round
        self.round_counter = 1

        self.total_sample_num = 0
        self.quit_sign = False
        self.theta = self._set_theta()

        # batch size: the number of words in total that are sent to the server;
        # check Corollary 1 in the paper.
        # Done in _set_theta: We need to make sure theta >= np.e ** (self.epsilon/self.MAX_L) - 1
        logging.info(f"train_data_num={train_data_num}")
        logging.info(f"self.epsilon = {self.epsilon}")
        logging.info(f"self.theta = {self.theta}")
        logging.info(f"self.MAX_L={self.MAX_L}")

        self.batch_size = int(train_data_num * (np.e ** (self.epsilon / self.MAX_L) - 1) / (
                self.theta * np.e ** (self.epsilon / self.MAX_L)))
        self.init_msg = int(math.ceil(self.batch_size * 1.0 / args.client_num_per_round))
        self.w_global = {}  # self.trie = {}

    def get_init_msg(self):
        return self.init_msg

    def set_init_msg(self, init_msg):
        self.init_msg = init_msg

    def aggregate(self, local_submission_list: List[Tuple[float, Any]]):
        votes = {}
        for (num, local_vote_dict) in local_submission_list:
            for key in local_vote_dict.keys():
                if key in votes:
                    votes[key] += local_vote_dict[key]
                else:
                    votes[key] = local_vote_dict[key]
        logging.info(f"aggregator ===================== votes = {votes}")

        if self.quit_sign or self.round_counter > self.MAX_L:
            print("end of discovery")
            self.print_heavy_hitters()
        else:
            self.server_update(votes)
            self.round_counter += 1
        return self.w_global

    def _set_theta(self):
        theta = 5  # initial guess
        delta_inverse = 1 / self.delta
        while ((theta - 3) / (theta - 2)) * math.factorial(theta) < delta_inverse:
            theta += 1
        while theta < np.e ** (self.epsilon / self.MAX_L) - 1:
            theta += 1
        print(f'Theta used by TrieHH: {theta}')
        return theta

    def server_update(self, votes):
        # It might make more sense to define a small class called server_state
        # server_state can track 2 things: 1) updated trie, and 2) quit_sign
        # server_state can be initialized in the constructor of SimulateTrieHH
        # and server_update would just update server_state
        # (i.e, it would update self.server_state.trie & self.server_state.quit_sign)
        self.quit_sign = True
        for prefix in votes:
            if votes[prefix] >= self.theta:
                self.w_global[prefix] = None
                self.quit_sign = False

    def print_heavy_hitters(self):
        heavy_hitters = []
        print(f"self.w_global = {self.w_global}")
        raw_result = self.w_global.keys()
        for word in raw_result:
            if word[-1:] == '$':
                heavy_hitters.append(word.rstrip('$'))
        # print(f'Discovered {len(heavy_hitters)} heavy hitters in run #{self.round_counter + 1}: {heavy_hitters}')
        print(f'Discovered {len(heavy_hitters)} heavy hitters: {heavy_hitters}')
