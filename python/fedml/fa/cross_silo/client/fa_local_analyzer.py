import time
from fedml.core.mlops import MLOpsProfilerEvent


class FALocalAnalyzer(object):
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        args,
        local_analyzer,
    ):
        self.local_analyzer = local_analyzer
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None
        self.args = args
        self.init_msg = None

    def set_init_msg(self, init_msg):
        self.local_analyzer.set_init_msg(init_msg)

    def get_init_msg(self):
        return self.local_analyzer.get_init_msg()

    def set_server_data(self, server_data):
        self.local_analyzer.set_server_data(server_data)

    def set_client_submission(self, client_submission):
        self.local_analyzer.set_client_submission(client_submission)

    def update_dataset(self, client_index):
        self.client_index = client_index

        if self.train_data_local_dict is not None:
            self.train_local = self.train_data_local_dict[client_index]
        else:
            self.train_local = None

        if self.train_data_local_num_dict is not None:
            self.local_sample_number = self.train_data_local_num_dict[client_index]
        else:
            self.local_sample_number = 0

        self.local_analyzer.update_dataset(self.train_local, self.local_sample_number)

    def local_analyze(self, round_idx=None):
        self.args.round_idx = round_idx
        tick = time.time()
        self.local_analyzer.local_analyze(self.train_local, self.args)

        MLOpsProfilerEvent.log_to_wandb({"Train/Time": time.time() - tick, "round": round_idx})
        client_submission = self.local_analyzer.get_client_submission()
        return client_submission, self.local_sample_number