import csv
import os


from fedml.data.fednlp.base.raw_data.base_raw_data_loader import (
    TextClassificationRawDataLoader,
)


# TODO: only used for creating the h5 files. not referenced by DataManager.
class RawDataLoader(TextClassificationRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_path = "train.csv"
        self.test_path = "test.csv"

    def load_data(self):
        if (
            len(self.X) == 0
            or len(self.Y) == 0
            or self.attributes["label_vocab"] is None
        ):
            train_size = self.process_data_file(
                os.path.join(self.data_path, self.train_path)
            )
            test_size = self.process_data_file(
                os.path.join(self.data_path, self.test_path)
            )
            self.attributes["label_vocab"] = {
                label: i for i, label in enumerate(set(self.Y.values()))
            }
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [
                i for i in range(train_size, train_size + test_size)
            ]
            self.attributes["index_list"] = (
                self.attributes["train_index_list"] + self.attributes["test_index_list"]
            )

    def process_data_file(self, file_path):
        cnt = 0
        with open(file_path, "r", newline="") as csvfile:
            data = csv.reader(csvfile, delimiter=",")
            for line in data:
                target = line[0]
                source = line[2].replace("\\", "")
                assert len(self.X) == len(self.Y)
                idx = len(self.X)
                self.X[idx] = source
                self.Y[idx] = target
                cnt += 1
        return cnt
