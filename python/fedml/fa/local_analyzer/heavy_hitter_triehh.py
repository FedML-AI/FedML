import math
import numpy as np
from collections import defaultdict
from fedml.fa.base_frame.client_analyzer import FAClientAnalyzer


class TrieHHClientAnalyzer(FAClientAnalyzer):
    def __init__(self, args):
        super().__init__(args=args)
        self.round_counter = 0
        self.batch_size = -1
        self.client_num_per_round = args.client_num_per_round

    def set_init_msg(self, init_msg):
        self.init_msg = init_msg
        self.batch_size = self.init_msg

    def get_init_msg(self):
        return self.init_msg

    def local_analyze(self, train_data, args):
        idxs = np.random.choice(range(len(train_data)), self.batch_size, replace=False)
        sample_local_dataset = [train_data[i] for i in idxs]
        votes = self.client_vote(sample_local_dataset)
        self.set_client_submission(votes)

    def client_vote(self, sample_local_dataset):
        votes = defaultdict(int)
        self.round_counter += 1
        self.w_global = self.get_server_data()
        for word in sample_local_dataset:
            vote_result = self.one_word_vote(word)
            if vote_result > 0:
                votes[word[0:self.round_counter]] += vote_result
        return votes

    def one_word_vote(self, word):
        if len(word) < self.round_counter:
            return 0
        pre = word[0:self.round_counter - 1]
        # print(f"self.w_global={self.w_global}")
        # print(f"pre = {pre}, type={type(self.w_global)}")
        if self.w_global is None:
            return 1
        if pre and (pre not in self.w_global):
            return 0
        return 1