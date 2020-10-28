import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.constants import *

train_file_path = "../../../../data/fednlp/sequence_tagging/wikigold/wikigold/CONLL-format/data/wikigold.conll.txt"


class DataLoader(BaseDataLoader):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        allowed_keys = {"padding", "max_sequence_length"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)

        self.sequence_length = []
        self.token_vocab = dict()
        self.label_vocab = dict()
        if self.padding:
            self.token_vocab[PAD_TOKEN] = len(self.token_vocab)
            self.label_vocab[PAD_LABEL] = len(self.label_vocab)

    def data_loader(self):
        self.process_data(self.data_path)

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

    def process_data(self, file_path):
        single_x = []
        single_y = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split(" ")
                    single_x.append(token)
                    single_y.append(label)
                else:
                    if len(single_x) != 0:
                        self.X.append(single_x.copy())
                        self.Y.append(single_y.copy())
                        self.sequence_length.append(len(single_x))
                    single_x.clear()
                    single_y.clear()


if __name__ == "__main__":
    data_loader = DataLoader(train_file_path)
    train_data_loader = data_loader.data_loader()
    print("done")
