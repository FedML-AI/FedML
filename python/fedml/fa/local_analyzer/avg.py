from fedml.fa.base_frame.client_analyzer import FAClientAnalyzer

class AverageClientAnalyzer(FAClientAnalyzer):
    """
    A client analyzer for calculating the average of values in the training data.

    Args:
        None

    Methods:
        local_analyze(train_data, args):
            Analyze the training data to calculate the average of values and set the client submission.

    """

    def local_analyze(self, train_data, args):
        """
        Analyze the training data to calculate the average of values and set the client submission.

        Args:
            train_data (list): The training data containing values to analyze.
            args: Additional arguments (not used in this method).

        Returns:
            None
        """
        sample_num = len(train_data)
        average = 0.0
        for value in train_data:
            average += float(value) / float(sample_num)
        self.set_client_submission(average)
