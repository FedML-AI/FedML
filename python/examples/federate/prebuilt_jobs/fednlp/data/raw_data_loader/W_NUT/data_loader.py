import os

from fedml.data.fednlp.base.raw_data.base_raw_data_loader import SeqTaggingRawDataLoader


class RawDataLoader(SeqTaggingRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_file_name = "train_data/Conll_Format"
        self.test_file_name = "test_data_2020/Conll_Format"

    def load_data(self):
        if (
            len(self.X) == 0
            or len(self.Y) == 0
            or self.attributes["label_vocab"] is None
        ):
            train_dir = os.path.join(self.data_path, self.train_file_name)
            train_size = 0
            for root, _, files in os.walk(train_dir):
                for file_name in files:
                    file_path = os.path.join(train_dir, file_name)
                    train_size += self.process_data_file(file_path)
            test_dir = os.path.join(self.data_path, self.test_file_name)
            test_size = 0
            for root, _, files in os.walk(test_dir):
                for file_name in files:
                    file_path = os.path.join(test_dir, file_name)
                    test_size += self.process_data_file(file_path)
            # train_size = self.process_data_file(os.path.join(self.data_path, self.train_file_name))
            # test_size = self.process_data_file(os.path.join(self.data_path, self.test_file_name))
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [
                i for i in range(train_size, train_size + test_size)
            ]
            self.attributes["index_list"] = (
                self.attributes["train_index_list"] + self.attributes["test_index_list"]
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
                    token, label = line.split("\t")
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
