import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.partition import *
from base.utils import *
class DataLoder(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "sequence_tagging"
        self.target_vocab = None

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or len(self.target_vocab) == 0:
            X, Y = self.process_data(self.data_path)
            self.X, self.Y = X, Y
            self.target_vocab = build_vocab(Y)

        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type}

    def process_data(self, file_path):
        X = []
        Y = []
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
                    if len(single_x) != 0 and len(single_y) == len(single_y):
                        X.append(single_x.copy())
                        Y.append(single_y.copy())
                    single_x.clear()
                    single_y.clear()
        return X, Y



if __name__ == "__main__":
    import pickle
    train_file_path = "../../../../data/fednlp/sequence_tagging/wikigold/wikigold/CONLL-format/data/wikigold.conll.txt"
    train_data_loader = DataLoder(train_file_path)
    train_result = train_data_loader.data_loader()
    uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]])
    # pickle.dump(train_result, open("wikigold_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("wikigold_partition.pkl", "wb"))
    print("done")
