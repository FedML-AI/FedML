import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *


class DataLoader(BaseDataLoader):
    def __init__(self, data_path, partition, **kwargs):
        super().__init__(data_path, partition, **kwargs)
        allowed_keys = {"padding", "max_sequence_length"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)

        if callable(self.partition):
            X, Y, _ = self.process_data(self.data_path)
            self.attributes = self.partition(X, Y)
        else:
            self.attributes = self.process_attributes()

        self.sequence_length = []
        self.token_vocab = dict()
        self.label_vocab = dict()
        if self.padding:
            self.token_vocab[PAD_TOKEN] = len(self.token_vocab)
            self.label_vocab[PAD_LABEL] = len(self.label_vocab)

    def data_loader(self, client_idx=None):
        if client_idx is not None:
            X, Y, sequence_length = self.process_data(self.data_path, client_idx)
        else:
            X, Y, sequence_length = self.process_data(self.data_path)

        self.X, self.Y, self.sequence_length = X, Y, sequence_length

        result = dict()

        self.build_vocab(self.X, self.token_vocab)
        self.build_vocab(self.Y, self.label_vocab)

        result["token_vocab"] = self.token_vocab
        result["label_vocab"] = self.label_vocab
        if self.padding:
            if not self.max_sequence_length:
                self.max_sequence_length = max(self.sequence_length)
            self.padding_data(self.X, self.max_sequence_length, False)
            self.padding_data(self.Y, self.max_sequence_length, False, PAD_LABEL)
            result["max_sequence_length"] = self.max_sequence_length
        result["sequence_length"] = self.sequence_length
        result["X"] = self.X
        result["Y"] = self.Y
        return result

    def process_data(self, file_path, client_idx=None):
        X = []
        Y = []
        sequence_length = []
        single_x = []
        single_y = []
        cnt = 0
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split(" ")
                    single_x.append(token)
                    single_y.append(label)
                else:
                    if len(single_x) != 0:
                        if client_idx is not None and client_idx != self.attributes["inputs"][cnt]:
                            cnt += 1
                            continue
                        X.append(single_x.copy())
                        Y.append(single_y.copy())
                        sequence_length.append(len(single_x))
                        cnt += 1
                    single_x.clear()
                    single_y.clear()
        return X, Y, sequence_length

def test_performance():
    import time
    from pympler import asizeof
    train_file_path = "../../../../data/fednlp/sequence_tagging/wikigold/wikigold/CONLL-format/data/wikigold.conll.txt"
    # load all data
    start = time.time()
    data_loader = DataLoader(train_file_path, uniform_partition, padding=True)
    train_data_loader = data_loader.data_loader()
    end = time.time()
    print("all data:", end - start)
    print("size:", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))
    # load a part of data
    start = time.time()
    data_loader = DataLoader(train_file_path, uniform_partition, padding=True)
    train_data_loader = data_loader.data_loader(0)
    end = time.time()
    print("part of data:", end - start)
    print("size:", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))



if __name__ == "__main__":
    test_performance()
