import os

from fedml.data.fednlp.base.raw_data.base_raw_data_loader import SeqTaggingRawDataLoader


class RawDataLoader(SeqTaggingRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)

    def load_data(self):
        if (
            len(self.X) == 0
            or len(self.Y) == 0
            or self.attributes["label_vocab"] is None
        ):
            train_size = 0
            test_size = 0
            train_source_list = list()
            test_source_list = list()
            for root, _, files in os.walk(self.data_path):
                for file_name in files:
                    file_path = os.path.join(self.data_path, file_name)
                    source = file_name.split("_")
                    if file_name.endswith("test.txt"):
                        size = self.process_data_file(file_path)
                        test_size += size
                        test_source_list.extend([source] * size)
                    elif file_name.endswith("train.txt") or file_name.endswith(
                        "dev.txt"
                    ):
                        size = self.process_data_file(file_path)
                        train_size += size
                        train_source_list.extend([source] * size)
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [
                i for i in range(train_size, train_size + test_size)
            ]
            self.attributes["index_list"] = (
                self.attributes["train_index_list"] + self.attributes["test_index_list"]
            )
            self.attributes["train_source_list"] = train_source_list
            self.attributes["test_source_list"] = test_source_list
            self.attributes["source_list"] = train_source_list + test_source_list

            assert len(self.attributes["train_index_list"]) == len(
                self.attributes["train_source_list"]
            )
            assert len(self.attributes["test_index_list"]) == len(
                self.attributes["test_source_list"]
            )
            assert len(self.attributes["index_list"]) == len(
                self.attributes["source_list"]
            )
            self.attributes["label_vocab"] = dict()
            for labels in self.Y.values():
                for label in labels:
                    if label not in self.attributes["label_vocab"]:
                        self.attributes["label_vocab"][label] = len(
                            self.attributes["label_vocab"]
                        )

    def process_data_file(self, file_path):
        single_x = []
        single_y = []
        cnt = 0
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split(" ")
                    single_x.append(token)
                    single_y.append(label)
                else:
                    if len(single_x) != 0:
                        assert len(self.X) == len(self.Y)
                        idx = len(self.X)
                        self.X[idx] = single_x.copy()
                        self.Y[idx] = single_y.copy()
                        cnt += 1
                    single_x.clear()
                    single_y.clear()
        return cnt
