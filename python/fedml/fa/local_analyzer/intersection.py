from fedml.fa.base_frame.client_analyzer import FAClientAnalyzer


class IntersectionClientAnalyzer(FAClientAnalyzer):
    """
    A client analyzer for finding the intersection of values in the training data.

    Args:
        None

    Methods:
        local_analyze(train_data, args):
            Analyze the training data to find the intersection of values and set the client submission.

    """

    def local_analyze(self, train_data, args):
        """
        Analyze the training data to find the intersection of values and set the client submission.

        Args:
            train_data (list): The training data containing values to analyze.
            args: Additional arguments (not used in this method).

        Returns:
            None
        """
        self.set_client_submission(list(set(train_data)))
