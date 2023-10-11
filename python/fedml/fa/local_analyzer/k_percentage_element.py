from fedml.fa.base_frame.client_analyzer import FAClientAnalyzer

class KPercentileElementClientAnalyzer(FAClientAnalyzer):
    """
    A client analyzer for counting values larger than a given percentile.

    Args:
        None

    Methods:
        local_analyze(train_data, args):
            Analyze the training data to count values larger than a given percentile and set the client submission.

    """

    def local_analyze(self, train_data, args):
        """
        Analyze the training data to count values larger than a given percentile and set the client submission.

        Args:
            train_data (list): The training data containing values to analyze.
            args: Additional arguments (not used in this method).

        Returns:
            None
        """
        counter = 0
        for data in train_data:
            if data >= self.server_data:  # flag
                counter += 1
        self.set_client_submission(counter)  # number of values that are larger than the flag
