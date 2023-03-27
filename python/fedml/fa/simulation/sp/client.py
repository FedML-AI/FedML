import numpy as np


class Client:
    def __init__(
            self, client_idx, local_training_data, local_datasize, args, local_analyzer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_datasize = local_datasize
        self.local_sample_number = 0
        self.args = args
        self.local_analyzer = local_analyzer

    def update_local_dataset(self, client_idx, local_training_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_sample_number = local_sample_number
        self.local_analyzer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def local_analyze(self, w_global):
        self.local_analyzer.set_server_data(w_global)
        idxs = np.random.choice(range(len(self.local_training_data)), self.local_sample_number, replace=False)
        train_data = [self.local_training_data[i] for i in idxs]
        # print(f"train data = {train_data}")
        self.local_analyzer.local_analyze(train_data, self.args)
        return self.local_analyzer.get_client_submission()
