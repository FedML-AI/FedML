import os
import re


from fedml.data.fednlp.base.raw_data.base_raw_data_loader import (
    TextClassificationRawDataLoader,
)


class RawDataLoader(TextClassificationRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_file_name = "SemEval2010_task8_training/TRAIN_FILE.TXT"
        self.test_file_name = "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

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
            self.target_vocab = {key: i for i, key in enumerate(set(self.Y.values()))}
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [
                i for i in range(train_size, train_size + test_size)
            ]
            self.attributes["index_list"] = (
                self.attributes["train_index_list"] + self.attributes["test_index_list"]
            )

    def process_data_file(self, file_path):
        cnt = 0
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            clean_data = None
            for i in range(len(data)):
                if len(data[i]) > 1 and data[i][0].isdigit():
                    clean_data = data[i].split("\t")[1][1:-1].strip()

                elif len(data[i - 1]) > 1 and data[i - 1][0].isdigit():
                    label = data[i].rstrip("\n")
                    assert len(self.X) == len(self.Y)
                    idx = len(self.X)
                    self.X[idx] = clean_data
                    self.Y[idx] = label
                    cnt += 1
        return cnt
