# use conll format to load the data
import os
import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *




class DataLoader(BaseDataLoader):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        allowed_keys = {"padding", "max_sequence_length"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)

        self.sequence_length = []
        self.token_vocab = dict()
        self.label_vocab = dict()
        self.attributes = dict()
        if self.padding:
            self.token_vocab[PAD_TOKEN] = len(self.token_vocab)
            self.label_vocab[PAD_LABEL] = len(self.label_vocab)

    def data_loader(self):
        self.attributes["inputs"] = []
        for file_path in self.data_path:
            self.process_data(file_path)

        result = dict()

        result["attributes"] = self.attributes

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
                    token, label = line.split("\t")
                    single_x.append(token)
                    single_y.append(label)
                else:
                    if len(single_x) != 0:
                        self.X.append(single_x.copy())
                        self.Y.append(single_y.copy())
                        self.sequence_length.append(len(single_x))
                        self.attributes["inputs"].append({"file_path": file_path})
                    single_x.clear()
                    single_y.clear()

    @staticmethod
    def partition(keys, values, attributes):
        file_path_dict = dict()
        for attribute in attributes["inputs"]:
            if attribute["file_path"] not in file_path_dict:
                file_path_dict[attribute["file_path"]] = len(file_path_dict)
        length = len(values[0])
        result = dict()
        for key in keys:
            result[key] = dict()
        for i in range(length):
            client_idx = file_path_dict[attributes["inputs"][i]["file_path"]]
            for j, key in enumerate(keys):
                if client_idx not in result[key]:
                    result[key][client_idx] = [values[j][i]]
                else:
                    result[key][client_idx].append(values[j][i])
                    result[key][client_idx].append(values[j][i])
        return result



if __name__ == "__main__":
    train_file_path = "../../../../data/fednlp/sequence_tagging/W-NUT 2017/data/train_data/Conll_Format/"
    dev_file_path = "../../../../data/fednlp/sequence_tagging/W-NUT 2017/data/dev_data/Conll_Format/"
    test_file_path = "../../../../data/fednlp/sequence_tagging/W-NUT 2017/data/test_data/Conll_Format/"
    test_2020_file_path = "../../../../data/fednlp/sequence_tagging/W-NUT 2017/data/test_data_2020/Conll_Format/"
    train_file_paths = [os.path.join(root, file_name) for root, dirs, files in os.walk(train_file_path)
                        for file_name in files]
    data_loader = DataLoader(train_file_paths)
    train_data_loader = data_loader.data_loader()
    partition(train_data_loader, method='uniform')
    print("done")
