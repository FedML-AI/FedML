from fedml.fa.base_frame.client_analyzer import FAClientAnalyzer


class KPercentileElementClientAnalyzer(FAClientAnalyzer):
    def local_analyze(self, train_data, args):
        counter = 0
        for data in train_data:
            if data >= self.server_data:  # flag
                counter += 1
        self.set_client_submission(counter)  # number of values that are larger than flag
