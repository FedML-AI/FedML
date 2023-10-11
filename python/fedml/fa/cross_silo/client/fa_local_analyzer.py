import time
from fedml.core.mlops import MLOpsProfilerEvent


class FALocalAnalyzer(object):
    """
    A class representing a local analyzer for federated learning.

    Args:
        client_index (int): The index of the client.
        train_data_local_dict (dict): A dictionary containing local training data.
        train_data_local_num_dict (dict): A dictionary containing the number of local training samples.
        train_data_num (int): The total number of training samples.
        args: Configuration arguments.
        local_analyzer: An instance of the local analyzer.

    Attributes:
        local_analyzer: An instance of the local analyzer.
        client_index (int): The index of the client.
        train_data_local_dict (dict): A dictionary containing local training data.
        train_data_local_num_dict (dict): A dictionary containing the number of local training samples.
        all_train_data_num (int): The total number of training samples.
        train_local: Local training data for the client.
        local_sample_number: The number of local training samples for the client.
        test_local: Local testing data for the client.
        args: Configuration arguments.
        init_msg: Initialization message for the client.

    Methods:
        set_init_msg(init_msg):
            Set the initialization message for the client.

        get_init_msg():
            Get the initialization message for the client.

        set_server_data(server_data):
            Set the server data for the client.

        set_client_submission(client_submission):
            Set the client's submission.

        update_dataset(client_index):
            Update the client's dataset based on the provided client index.

        local_analyze(round_idx=None):
            Perform local analysis for federated learning.

    """
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        args,
        local_analyzer,
    ):
        """
        Initialize the FALocalAnalyzer.

        Args:
            client_index (int): The index of the client.
            train_data_local_dict (dict): A dictionary containing local training data.
            train_data_local_num_dict (dict): A dictionary containing the number of local training samples.
            train_data_num (int): The total number of training samples.
            args: Configuration arguments.
            local_analyzer: An instance of the local analyzer.

        Returns:
            None
        """
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
        """
        Set the initialization message for the client.

        Args:
            init_msg: Initialization message for the client.

        Returns:
            None
        """
        self.local_analyzer.set_init_msg(init_msg)

    def get_init_msg(self):
        """
        Get the initialization message for the client.

        Returns:
            Initialization message for the client.
        """
        return self.local_analyzer.get_init_msg()

    def set_server_data(self, server_data):
        """
        Set the server data for the client.

        Args:
            server_data: Server data for the client.

        Returns:
            None
        """
        self.local_analyzer.set_server_data(server_data)

    def set_client_submission(self, client_submission):
        """
        Set the client's submission.

        Args:
            client_submission: Client's submission data.

        Returns:
            None
        """
        self.local_analyzer.set_client_submission(client_submission)

    def update_dataset(self, client_index):
        """
        Update the client's dataset based on the provided client index.

        Args:
            client_index (int): The index of the client.

        Returns:
            None
        """
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
        """
        Perform local analysis for federated learning.

        Args:
            round_idx (int): The current round index (default is None).

        Returns:
            Tuple containing client submission data and the number of local samples.
        """
        self.args.round_idx = round_idx
        tick = time.time()
        self.local_analyzer.local_analyze(self.train_local, self.args)

        MLOpsProfilerEvent.log_to_wandb({"Train/Time": time.time() - tick, "round": round_idx})
        client_submission = self.local_analyzer.get_client_submission()
        return client_submission, self.local_sample_number
