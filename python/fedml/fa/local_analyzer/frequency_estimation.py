from fedml.fa.base_frame.client_analyzer import FAClientAnalyzer


class FrequencyEstimationClientAnalyzer(FAClientAnalyzer):
    """
    A client analyzer for estimating the frequency of values in the training data.

    Args:
        client_id: The unique identifier of the client.
        server: The federated learning server.

    Attributes:
        client_id: The unique identifier of the client.
        server: The federated learning server.

    Methods:
        local_analyze(train_data, args):
            Analyze the training data to estimate the frequency of values and set the client submission.

    """

    def local_analyze(self, train_data, args):
        """
        Analyze the training data to estimate the frequency of values and set the client submission.

        Args:
            train_data (list): The training data containing values to analyze.
            args: Additional arguments (not used in this method).

        Returns:
            None
        """
        counter_dict = dict()

        for value in train_data:
            if counter_dict.get(value) is None:
                counter_dict[value] = 1
            else:
                counter_dict[value] += 1

        self.set_client_submission(counter_dict)
