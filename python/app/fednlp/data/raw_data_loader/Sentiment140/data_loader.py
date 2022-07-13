import csv
import os
import re
import string


from fedml.data.fednlp.base.raw_data.base_raw_data_loader import (
    TextClassificationRawDataLoader,
)


class RawDataLoader(TextClassificationRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.test_file_name = "testdata.manual.2009.06.14.csv"
        self.train_file_name = "training.1600000.processed.noemoticon.csv"

    def load_data(self):
        if (
            len(self.X) == 0
            or len(self.Y) == 0
            or self.attributes["label_vocab"] is None
        ):
            train_size = self.process_data_file(
                os.path.join(self.data_path, self.train_file_name)
            )
            test_size = self.process_data_file(
                os.path.join(self.data_path, self.test_file_name)
            )
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [
                i for i in range(train_size, train_size + test_size)
            ]
            self.attributes["index_list"] = (
                self.attributes["train_index_list"] + self.attributes["test_index_list"]
            )
            self.attributes["label_vocab"] = {
                label: i for i, label in enumerate(set(self.Y.values()))
            }

    def process_data_file(self, file_path):
        cnt = 0
        with open(
            file_path, "r", newline="", encoding="utf-8", errors="ignore"
        ) as csvfile:
            data = csv.reader(csvfile, delimiter=",")
            for line in data:
                assert len(self.X) == len(self.Y)
                idx = len(self.X)
                self.X[idx] = line[5]
                if line[0] == "0":
                    self.Y[idx] = line[0]
                else:
                    self.Y[idx] = "1"
                cnt += 1

        return cnt
