import math
import numpy as np
from collections import defaultdict
from fedml.fa.base_frame.client_analyzer import FAClientAnalyzer

class TrieHHClientAnalyzer(FAClientAnalyzer):
    """
    A client analyzer for Trie-HH federated learning.

    Args:
        args: Additional arguments for configuration.

    Attributes:
        round_counter (int): Counter to keep track of rounds.
        batch_size (int): Size of the sample batch for analysis.
        client_num_per_round (int): Number of clients per round.

    Methods:
        __init__(self, args):
            Initialize the TrieHHClientAnalyzer with provided arguments.

        set_init_msg(self, init_msg):
            Set the initial message containing batch size.

        get_init_msg(self):
            Get the initial message.

        local_analyze(self, train_data, args):
            Analyze the local training data and set the client submission.

        client_vote(self, sample_local_dataset):
            Perform voting based on local data and return the votes.

        one_word_vote(self, word):
            Perform voting for a single word in the dataset.

    """

    def __init__(self, args):
        super().__init__(args=args)
        self.round_counter = 0
        self.batch_size = -1
        self.client_num_per_round = args.client_num_per_round

    def set_init_msg(self, init_msg):
        """
        Set the initial message containing batch size.

        Args:
            init_msg: The initial message containing batch size.

        Returns:
            None
        """
        self.init_msg = init_msg
        self.batch_size = self.init_msg

    def get_init_msg(self):
        """
        Get the initial message.

        Returns:
            int: The initial message containing batch size.
        """
        return self.init_msg

    def local_analyze(self, train_data, args):
        """
        Analyze the training data and set the client submission.

        Args:
            train_data (list): The training data for analysis.
            args: Additional arguments (not used in this method).

        Returns:
            None
        """
        idxs = np.random.choice(range(len(train_data)), self.batch_size, replace=False)
        sample_local_dataset = [train_data[i] for i in idxs]
        votes = self.client_vote(sample_local_dataset)
        self.set_client_submission(votes)

    def client_vote(self, sample_local_dataset):
        """
        Perform voting based on local data and return the votes.

        Args:
            sample_local_dataset (list): Sampled local dataset for voting.

        Returns:
            dict: Dictionary containing votes.
        """
        votes = defaultdict(int)
        self.round_counter += 1
        self.w_global = self.get_server_data()
        for word in sample_local_dataset:
            vote_result = self.one_word_vote(word)
            if vote_result > 0:
                votes[word[0:self.round_counter]] += vote_result
        return votes

    def one_word_vote(self, word):
        """
        Perform voting for a single word in the dataset.

        Args:
            word (str): A word from the dataset.

        Returns:
            int: Voting result (1 if valid, 0 otherwise).
        """
        if len(word) < self.round_counter:
            return 0
        pre = word[0:self.round_counter - 1]
        
        if self.w_global is None:
            return 1
        if pre and (pre not in self.w_global):
            return 0
        return 1
