import logging
from .fa_local_analyzer import FALocalAnalyzer
from ...local_analyzer.client_analyzer_creator import create_local_analyzer


class TrainerDistAdapter:
    """
    Adapter for a Federated Learning Trainer with Distributed Training.

    Args:
        args (object): An object containing trainer configuration parameters.
        client_rank (int): The rank of the client.
        train_data_num (int): The total number of training data samples.
        train_data_local_num_dict (dict): A dictionary of client-specific training data sizes.
        train_data_local_dict (dict): A dictionary of client-specific training data.
        local_analyzer: An instance of the local analyzer (optional).

    Attributes:
        client_index (int): The index of the client.
        client_rank (int): The rank of the client.
        local_analyzer: An instance of the local analyzer.
        args (object): An object containing trainer configuration parameters.

    Methods:
        local_analyze(round_idx):
            Perform local analysis for a given training round.
        set_server_data(server_data):
            Set server data for the local analyzer.
        set_init_msg(init_msg):
            Set initialization message for the local analyzer.
        set_client_submission(client_submission):
            Set client submission for the local analyzer.
        update_dataset(client_index=None):
            Update the dataset for the local analyzer.

    """
    def __init__(
        self,
        args,
        client_rank,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        local_analyzer,
    ):
        """
        Initialize the TrainerDistAdapter.

        Args:
            args (object): An object containing trainer configuration parameters.
            client_rank (int): The rank of the client.
            train_data_num (int): The total number of training data samples.
            train_data_local_num_dict (dict): A dictionary of client-specific training data sizes.
            train_data_local_dict (dict): A dictionary of client-specific training data.
            local_analyzer: An instance of the local analyzer (optional).

        Note:
            This constructor sets up the adapter and initializes it with the provided dataset and configuration.

        Returns:
            None
        """
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
        """
        Get an instance of the local analyzer.

        Args:
            client_index (int): The index of the client.
            train_data_local_dict (dict): A dictionary of client-specific training data.
            train_data_local_num_dict (dict): A dictionary of client-specific training data sizes.
            train_data_num (int): The total number of training data samples.
            args (object): An object containing trainer configuration parameters.
            local_analyzer: An instance of the local analyzer (optional).

        Returns:
            FALocalAnalyzer: An instance of the local analyzer.
        """
        return FALocalAnalyzer(
            client_index,
            train_data_local_dict,
            train_data_local_num_dict,
            train_data_num,
            args,
            local_analyzer,
        )

    def local_analyze(self, round_idx):
        """
        Perform local analysis for a given training round.

        Args:
            round_idx (int): The index of the training round.

        Returns:
            tuple: A tuple containing client submission and local sample count.
        """
        client_submission, local_sample_num = self.local_analyzer.local_analyze(round_idx)
        return client_submission, local_sample_num

    def set_server_data(self, server_data):
        """
        Set server data for the local analyzer.

        Args:
            server_data: Data received from the server.

        Returns:
            None
        """
        self.local_analyzer.set_server_data(server_data)

    def set_init_msg(self, init_msg):
        """
        Set initialization message for the local analyzer.

        Args:
            init_msg: Initialization message received from the server.

        Returns:
            None
        """
        self.local_analyzer.set_init_msg(init_msg)

    def set_client_submission(self, client_submission):
        """
        Set client submission for the local analyzer.

        Args:
            client_submission: Client's training submission.

        Returns:
            None
        """
        self.local_analyzer.set_client_submission(client_submission)

    def update_dataset(self, client_index=None):
        """
        Update the dataset for the local analyzer.

        Args:
            client_index (int): The index of the client (optional).

        Returns:
            None
        """
        _client_index = client_index or self.client_index
        self.local_analyzer.update_dataset(int(_client_index))
