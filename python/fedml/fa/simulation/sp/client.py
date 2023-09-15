import numpy as np

class Client:
    """
    Client class for Federated Analytics simulation.

    Args:
        client_idx (int): Index of the client.
        local_training_data (list): Local training data for the client.
        local_datasize (int): Size of the local training data.
        args (object): Arguments for the simulation.
        local_analyzer (object): Local analyzer instance.

    Attributes:
        client_idx (int): Index of the client.
        local_training_data (list): Local training data for the client.
        local_datasize (int): Size of the local training data.
        local_sample_number (int): Number of local samples.
        args (object): Arguments for the simulation.
        local_analyzer (object): Local analyzer instance.

    Methods:
        update_local_dataset(client_idx, local_training_data, local_sample_number):
            Update the client's local dataset and sample number.
        
        get_sample_number():
            Get the number of local samples.
        
        local_analyze(w_global):
            Perform local analysis and return client's submission.
    """

    def __init__(
            self, client_idx, local_training_data, local_datasize, args, local_analyzer,
    ):
        """
        Initialize the Client class.

        Args:
            client_idx (int): Index of the client.
            local_training_data (list): Local training data for the client.
            local_datasize (int): Size of the local training data.
            args (object): Arguments for the simulation.
            local_analyzer (object): Local analyzer instance.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_datasize = local_datasize
        self.local_sample_number = 0
        self.args = args
        self.local_analyzer = local_analyzer

    def update_local_dataset(self, client_idx, local_training_data, local_sample_number):
        """
        Update the client's local dataset and sample number.

        Args:
            client_idx (int): Index of the client.
            local_training_data (list): Updated local training data.
            local_sample_number (int): Updated number of local samples.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_sample_number = local_sample_number
        self.local_analyzer.set_id(client_idx)

    def get_sample_number(self):
        """
        Get the number of local samples.

        Returns:
            int: Number of local samples.
        """
        return self.local_sample_number

    def local_analyze(self, w_global):
        """
        Perform local analysis and return client's submission.

        Args:
            w_global (object): Global data from the server.

        Returns:
            object: Client's submission after local analysis.
        """
        self.local_analyzer.set_server_data(w_global)
        idxs = np.random.choice(range(len(self.local_training_data)), self.local_sample_number, replace=False)
        train_data = [self.local_training_data[i] for i in idxs]
        
        self.local_analyzer.local_analyze(train_data, self.args)
        return self.local_analyzer.get_client_submission()
