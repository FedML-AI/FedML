from ..common.singleton import Singleton


class MLOpsStatus(Singleton):
    _status_instance = None

    def __init__(self):
        """
        Initialize an instance of MLOpsStatus.

        This class is a Singleton and should not be instantiated directly.
        Use the `get_instance` method to obtain the Singleton instance.

        Attributes:
            messenger: Messenger object for communication.
            run_id: The ID of the current run.
            edge_id: The ID of the edge device.
            client_agent_status: A dictionary to store client agent status.
            server_agent_status: A dictionary to store server agent status.
            client_status: A dictionary to store client status.
            server_status: A dictionary to store server status.
        """
        self.messenger = None
        self.run_id = None
        self.edge_id = None
        self.client_agent_status = dict()
        self.server_agent_status = dict()
        self.client_status = dict()
        self.server_status = dict()

    @staticmethod
    def get_instance():
        """
        Get the Singleton instance of MLOpsStatus.

        Returns:
            MLOpsStatus: The Singleton instance of MLOpsStatus.
        """
        if MLOpsStatus._status_instance is None:
            MLOpsStatus._status_instance = MLOpsStatus()

        return MLOpsStatus._status_instance

    def set_client_agent_status(self, edge_id, status):
        """
        Set the status of a client agent.

        Args:
            edge_id (int): The ID of the edge device.
            status (str): The status of the client agent.
        """
        self.client_agent_status[edge_id] = status

    def set_server_agent_status(self, edge_id, status):
        """
        Set the status of a server agent.

        Args:
            edge_id (int): The ID of the edge device.
            status (str): The status of the server agent.
        """
        self.server_agent_status[edge_id] = status

    def set_client_status(self, edge_id, status):
        """
        Set the status of a client.

        Args:
            edge_id (int): The ID of the edge device.
            status (str): The status of the client.
        """
        self.client_status[edge_id] = status

    def set_server_status(self, edge_id, status):
        """
        Set the status of a server.

        Args:
            edge_id (int): The ID of the edge device.
            status (str): The status of the server.
        """
        self.server_status[edge_id] = status

    def get_client_agent_status(self, edge_id):
        """
        Get the status of a client agent.

        Args:
            edge_id (int): The ID of the edge device.

        Returns:
            str or None: The status of the client agent, or None if not found.
        """
        return self.client_agent_status.get(edge_id, None)

    def get_server_agent_status(self, edge_id):
        """
        Get the status of a server agent.

        Args:
            edge_id (int): The ID of the edge device.

        Returns:
            str or None: The status of the server agent, or None if not found.
        """
        return self.server_agent_status.get(edge_id, None)

    def get_client_status(self, edge_id):
        """
        Get the status of a client.

        Args:
            edge_id (int): The ID of the edge device.

        Returns:
            str or None: The status of the client, or None if not found.
        """
        return self.client_status.get(edge_id, None)

    def get_server_status(self, edge_id):
        """
        Get the status of a server.

        Args:
            edge_id (int): The ID of the edge device.

        Returns:
            str or None: The status of the server, or None if not found.
        """
        return self.server_status.get(edge_id, None)
