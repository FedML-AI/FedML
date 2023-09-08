import logging


class BaseCentralWorker(object):
    """
    Base class representing a central worker in a distributed system.

    This class is responsible for managing client local results and aggregating them.

    Attributes:
        client_num (int): The number of client processes.
        args (object): An object containing configuration parameters.
        client_local_result_list (dict): A dictionary to store client local results.
        flag_client_model_uploaded_dict (dict): A dictionary to track whether each client has uploaded results.

    Methods:
        add_client_local_result(index, client_local_result):
            Add client's local result to the worker.
        check_whether_all_receive():
            Check if all clients have uploaded their local results.
        aggregate():
            Aggregate client local results.
    """
    def __init__(self, client_num, args):
        """
        Initialize the BaseCentralWorker.

        Args:
            client_num (int): The number of client processes.
            args (object): An object containing configuration parameters.

        Returns:
            None
        """
        self.client_num = client_num
        self.args = args

        self.client_local_result_list = dict()

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def add_client_local_result(self, index, client_local_result):
        """
        Add client's local result to the worker.

        Args:
            index (int): The index of the client.
            client_local_result (object): The local result from the client.

        Returns:
            None
        """
        logging.info("add_client_local_result. index = %d" % index)
        self.client_local_result_list[index] = client_local_result
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        """
        Check if all clients have uploaded their local results.

        Returns:
            bool: True if all clients have uploaded their results, False otherwise.
        """
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        """
        Aggregate client local results.

        Returns:
            object: The aggregated global result.
        """
        global_result = 0
        for k in self.client_local_result_list.keys():
            global_result += self.client_local_result_list[k]
        return global_result
