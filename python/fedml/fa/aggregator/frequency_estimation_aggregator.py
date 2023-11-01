from matplotlib import pyplot as plt
from typing import List, Tuple, Any
from fedml.fa.base_frame.server_aggregator import FAServerAggregator


class FrequencyEstimationAggregatorFA(FAServerAggregator):
    def __init__(self, args):
        super().__init__(args)
        self.total_sample_num = 0
        self.set_server_data(server_data=[])
        self.round_idx = 0
        self.total_round = args.comm_round

    def aggregate(self, local_submission_list: List[Tuple[float, Any]]):
        training_num = 0
        (sample_num, averaged_params) = local_submission_list[0]
        for i in range(0, len(local_submission_list)):
            local_sample_number, local_submission = local_submission_list[i]
            if len(self.server_data) == 0:
                self.server_data = local_submission
            else:
                for key in local_submission:
                    if self.server_data.get(key) is None:
                        self.server_data[key] = local_submission[key]
                    else:
                        self.server_data[key] = self.server_data[key] + local_submission[key]
            training_num += sample_num
        self.total_sample_num += training_num
        print(f"self.total_round={self.total_round}, round_idx={self.round_idx}, aggregation result = {self.server_data}")
        if self.round_idx == self.total_round - 1:
            self.print_frequency_estimation_results()
        self.round_idx += 1
        return self.server_data

    def print_frequency_estimation_results(self):
        print("frequency estimation: ")
        for key in self.server_data:
            print(f"key = {key}, freq = {self.server_data[key] / self.total_sample_num}")
        plt.bar(self.server_data.keys(), self.server_data.values(), align='center')
        plt.xlabel('Keys')
        plt.ylabel('Occurrence # ')
        plt.title('Histogram')
        plt.savefig('frequency_estimation_result.png', dpi=300, bbox_inches='tight')