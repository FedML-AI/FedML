from fedml.fa.base_frame.client_analyzer import FAClientAnalyzer


class AverageClientAnalyzer(FAClientAnalyzer):
    def local_analyze(self, train_data, args):
        sample_num = len(train_data)
        average = 0.0
        for value in train_data:
            average = average + float(value) / float(sample_num)
        self.set_client_submission(average)
