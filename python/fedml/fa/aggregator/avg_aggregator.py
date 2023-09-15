from typing import List, Tuple, Any
from fedml.fa.base_frame.server_aggregator import FAServerAggregator


class AVGAggregatorFA(FAServerAggregator):
    """
    Aggregator for Federated Learning with Averaging.

    Args:
        args (object): An object containing aggregator configuration parameters.

    Attributes:
        total_sample_num (int): The total number of training samples aggregated.
        server_data (float): The aggregated server data.

    Methods:
        aggregate(local_submission_list):
            Aggregate local submissions from clients and compute the weighted average.

    """
    def __init__(self, args):
        """
        Initialize the AVGAggregatorFA.

        Args:
            args (object): An object containing aggregator configuration parameters.

        Returns:
            None
        """
        super().__init__(args)
        self.total_sample_num = 0
        self.set_server_data(server_data=0)

    def aggregate(self, local_submission_list: List[Tuple[float, Any]]):
        """
        Aggregate local submissions from clients and compute the weighted average.

        Args:
            local_submission_list (list): A list of tuples containing local sample number and local submissions.

        Returns:
            float: The computed weighted average.
        """
        print(f"local_submission_list={local_submission_list}")
        training_num = 0
        for idx in range(len(local_submission_list)):
            (sample_num, local_submission) = local_submission_list[idx]
            training_num += sample_num

        avg = 0
        for i in range(0, len(local_submission_list)):
            local_sample_number, local_submission = local_submission_list[i]
            w = local_sample_number / training_num
            if i == 0:
                avg = local_submission * w
            else:
                avg += local_submission * w
        self.total_sample_num += training_num
        avg = avg * (training_num / self.total_sample_num) + self.server_data * (
                (self.total_sample_num - training_num) / self.total_sample_num)
        self.server_data = avg
        return avg



""" todo: 
Mode 1: (online mode) each client stores its AVG result and the total number of data being sampled so far; 
later computation will use this result.
Mode 2: (offline mode, no need to use iterations) clients do not store previous results; 
server collects results from clients and does a weighted avg each round.
Finally, server does a weighted avg for all rounds.
Mode 3: (online mode, server does not need to store avg results for each rounds, the clients do not store their answers) 
similar to fl, the server sends the AVG result & total sample num so far to each client; 
(or, AVG result + cdp && a fake total sample num, the server can do further computation to get the real answer)
Mode 4: (online mode) server sets 2 local var: avg and total sample num. The server collects answers from clients each round and compute AVG
using avg, total sample num, and training num of the current round
"""