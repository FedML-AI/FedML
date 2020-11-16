# use conll format to load the data
import os
import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.partition import *
from base.utils import *

class DataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "sequence_tagging"
        self.target_vocab = None

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X = None
            Y = None
            for root, dirs, files in os.walk(self.data_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if X is None or Y is None:
                        X, Y = self.process_data(file_path)
                    else:
                        temp = self.process_data(file_path)
                        X.extend(temp[0])
                        Y.extend(temp[1])
            self.X, self.Y = X, Y
            self.target_vocab = build_vocab(Y)

        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type}

    def process_data(self, file_path):
        X = []
        Y = []
        single_x = []
        single_y = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split("\t")
                    single_x.append(token)
                    single_y.append(label)
                else:
                    if len(single_x) != 0:
                        X.append(single_x.copy())
                        Y.append(single_y.copy())
                    single_x.clear()
                    single_y.clear()
        return X, Y

if __name__ == "__main__":
    import pickle
    train_file_path = "../../../../data/fednlp/sequence_tagging/W-NUT2017/data/train_data/Conll_Format/"
    dev_file_path = "../../../../data/fednlp/sequence_tagging/W-NUT2017/data/dev_data/Conll_Format/"
    test_file_path = "../../../../data/fednlp/sequence_tagging/W-NUT2017/data/test_data/Conll_Format/"
    test_2020_file_path = "../../../../data/fednlp/sequence_tagging/W-NUT2017/data/test_data_2020/Conll_Format/"
    train_data_loader = DataLoader(train_file_path)
    train_result = train_data_loader.data_loader()
    test_data_loader = DataLoader(test_file_path)
    test_result = test_data_loader.data_loader()
    uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]],
                                               [test_result["X"], test_result["Y"]])
    pickle_dict = train_result
    pickle_dict["X"].extend(test_result["X"])
    pickle_dict["Y"].extend(test_result["Y"])
    # pickle.dump(pickle_dict, open("w_nut_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("w_nut_partition.pkl", "wb"))
    print("done")
