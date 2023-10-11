import logging
from abc import ABC, abstractmethod


class FAClientAnalyzer(ABC):
    def __init__(self, args):
        """
        Initialize the client analyzer.

        Args:
            args: Configuration arguments.

        Returns:
            None
        """
        self.client_submission = 0
        self.id = 0
        self.args = args
        self.local_train_dataset = None
        self.server_data = None
        self.init_msg = None

    def set_init_msg(self, init_msg):
        """
        Set the initialization message.

        Args:
            init_msg: The initialization message.

        Returns:
            None
        """
        pass

    def get_init_msg(self):
        """
        Get the initialization message.

        Returns:
            Any: The initialization message.
        """
        pass

    def set_id(self, analyzer_id):
        """
        Set the ID of the client analyzer.

        Args:
            analyzer_id: The ID of the analyzer.

        Returns:
            None
        """
        self.id = analyzer_id

    def get_client_submission(self):
        """
        Get the client submission.

        Returns:
            Any: The client submission.
        """
        return self.client_submission

    def set_client_submission(self, client_submission):
        """
        Set the client submission.

        Args:
            client_submission: The client submission.

        Returns:
            None
        """
        self.client_submission = client_submission

    def get_server_data(self):
        """
        Get the server data.

        Returns:
            Any: The server data.
        """
        return self.server_data

    def set_server_data(self, server_data):
        """
        Set the server data.

        Args:
            server_data: The server data.

        Returns:
            None
        """
        self.server_data = server_data

    @abstractmethod
    def local_analyze(self, train_data, args):
        """
        Perform local analysis.

        Args:
            train_data: The local training data.
            args: Configuration arguments.

        Returns:
            None
        """
        pass

    def update_dataset(self, local_train_dataset, local_sample_number):
        """
        Update the local dataset.

        Args:
            local_train_dataset: The local training dataset.
            local_sample_number: The number of local samples.

        Returns:
            None
        """
        self.local_train_dataset = local_train_dataset
        self.local_sample_number = local_sample_number
