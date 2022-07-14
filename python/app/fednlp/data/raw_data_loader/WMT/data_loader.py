import pickle


from fedml.data.fednlp.base.raw_data.base_raw_data_loader import Seq2SeqRawDataLoader
from data.raw_data_loader.base.partition import *


class RawDataLoader(Seq2SeqRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)

    def load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0:
            total_size = self.process_data_file(self.data_path)
            self.attributes["index_list"] = [i for i in range(total_size)]

    def process_data_file(self, file_path):
        source_file_path = file_path[0]
        target_file_path = file_path[1]
        source_size = 0
        with open(source_file_path, "r") as f:
            for line in f:
                line = line.strip()
                idx = len(self.X)
                self.X[idx] = line
                source_size += 1
        target_size = 0
        with open(target_file_path, "r") as f:
            for line in f:
                line = line.strip()
                idx = len(self.Y)
                self.Y[idx] = line
                target_size += 1
        assert source_size == target_size
        return source_size
