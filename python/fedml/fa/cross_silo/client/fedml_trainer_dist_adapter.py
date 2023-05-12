import logging
from .fa_local_analyzer import FALocalAnalyzer
from ...local_analyzer.client_analyzer_creator import create_local_analyzer


class TrainerDistAdapter:
    def __init__(
        self,
        args,
        client_rank,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        local_analyzer,
    ):
        if local_analyzer is None:
            local_analyzer = create_local_analyzer(args=args)

        client_index = client_rank - 1
        local_analyzer.set_id(client_index)

        logging.info("Initiating Trainer")
        local_analyzer = self.get_local_analyzer(
            client_index,
            train_data_local_dict,
            train_data_local_num_dict,
            train_data_num,
            args,
            local_analyzer,
        )
        self.client_index = client_index
        self.client_rank = client_rank
        self.local_analyzer = local_analyzer
        self.args = args

    def get_local_analyzer(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        args,
        local_analyzer,
    ):
        return FALocalAnalyzer(
            client_index,
            train_data_local_dict,
            train_data_local_num_dict,
            train_data_num,
            args,
            local_analyzer,
        )

    def local_analyze(self, round_idx):
        client_submission, local_sample_num = self.local_analyzer.local_analyze(round_idx)
        return client_submission, local_sample_num

    def set_server_data(self, server_data):
        self.local_analyzer.set_server_data(server_data)

    def set_init_msg(self, init_msg):
        self.local_analyzer.set_init_msg(init_msg)

    def set_client_submission(self, client_submission):
        self.local_analyzer.set_client_submission(client_submission)

    def update_dataset(self, client_index=None):
        _client_index = client_index or self.client_index
        self.local_analyzer.update_dataset(int(_client_index))