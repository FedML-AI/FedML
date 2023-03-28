import logging
from typing import List, Tuple, Any
from fedml.fa.base_frame.server_aggregator import FAServerAggregator

"""
Step:
1. Set flag value, e.g., the initial value of the k percentile element. If the user has some information about the dataset,
e.g., knowing the average value, the user can pass the value as a parameter. Otherwise, the flag is set to 100
2. in iteration, the server sends the flag to each client to count the number of values that are larger than the flag.
If exactly = k%, done; otherwise, update the value of flag.

todo: median: http://proceedings.mlr.press/v80/yin18a/yin18a.pdf; geometric median

todo: do not converge..
"""


class KPercentileElementAggregatorFA(FAServerAggregator):
    def __init__(self, args, train_data_num):
        super().__init__(args)
        self.total_sample_num = 0
        self.set_server_data(server_data=[])
        self.quit = False
        self.total_sample_num = 0
        self.train_data_num_in_total = train_data_num
        self.percentage = args.k / 100
        if hasattr(args, "flag"):
            self.server_data = args.flag
            self.previous_server_data = args.flag
        else:
            self.server_data = 100
            self.previous_server_data = 100
        if hasattr(args, "use_all_data") and args.use_all_data in [False]:
            self.use_all_data = False  # in each iteration, each client randomly sample some data to compute
        else:
            self.use_all_data = True  # in each iteration, each client uses its all local data to compute
        self.max_val = self.previous_server_data
        self.min_val = self.previous_server_data

    def aggregate(self, local_submission_list: List[Tuple[float, Any]]):
        if self.quit:
            return self.server_data
        total_sample_num_this_round = 0
        local_satisfied_data_num_current_round = 0
        logging.info(f"flag={self.server_data}, local_submission_list={local_submission_list}")
        for (sample_num, satisfied_counter) in local_submission_list:
            total_sample_num_this_round += sample_num
            local_satisfied_data_num_current_round += satisfied_counter
        if total_sample_num_this_round * self.percentage == local_satisfied_data_num_current_round:
            self.quit = True
            self.previous_server_data = self.server_data
        elif total_sample_num_this_round * self.percentage > local_satisfied_data_num_current_round:
            # decrease server_data
            self.max_val = self.server_data
            if self.previous_server_data >= self.server_data:
                self.previous_server_data = self.server_data
                if self.server_data / 2 < self.min_val < self.max_val:
                    self.server_data = (self.server_data + self.min_val)/2
                else:
                    self.server_data = self.server_data / 2
                    self.min_val = self.server_data  # set lower bound for flag
            else:
                new_server_data = (self.previous_server_data + self.server_data) / 2
                self.previous_server_data = self.server_data
                self.server_data = new_server_data
        else:  # increase server_data
            self.min_val = self.server_data
            if self.previous_server_data <= self.server_data:
                self.previous_server_data = self.server_data
                if 2 * self.server_data > self.max_val > self.min_val:
                    self.server_data = (self.server_data + self.max_val) / 2
                else:
                    self.server_data = 2 * self.server_data
                    self.max_val = self.server_data
            else:
                new_server_data = (self.previous_server_data + self.server_data) / 2
                self.previous_server_data = self.server_data
                self.server_data = new_server_data
        return self.server_data
