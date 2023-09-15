from abc import ABC
from typing import List, Tuple, Any

class FAServerAggregator(ABC):
    def __init__(self, args):
        """
        Initialize the server aggregator.

        Args:
            args: Configuration arguments.

        Returns:
            None
        """
        self.id = 0
        self.args = args
        self.eval_data = None
        self.server_data = None
        self.init_msg = None

    def get_init_msg(self):
        """
        Get the initialization message.

        Returns:
            Any: The initialization message.
        """
        pass

    def set_init_msg(self, init_msg):
        """
        Set the initialization message.

        Args:
            init_msg: The initialization message.

        Returns:
            None
        """
        pass

    def set_id(self, aggregator_id):
        """
        Set the ID of the server aggregator.

        Args:
            aggregator_id: The ID of the aggregator.

        Returns:
            None
        """
        self.id = aggregator_id

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

    def aggregate(self, local_submissions: List[Tuple[float, Any]]):
        """
        Aggregate local submissions from clients.

        Args:
            local_submissions (List[Tuple[float, Any]]): A list of tuples containing local submissions and weights.

        Returns:
            None
        """
        pass
