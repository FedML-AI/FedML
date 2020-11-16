import os
import math
import random
import sys
import time
sys.path.append('..')

from base.data_loader import BaseDataLoader
from base.utils import *
from base.partition import *

class DataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "classification"
        self.target_vocab = None

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X = []
            Y = []
            for root1, dirs, _ in os.walk(self.data_path):
                for dir in dirs:
                    for root2, _, files in os.walk(os.path.join(root1, dir)):
                        for file_name in files:
                            file_path = os.path.join(root2, file_name)
                            X.extend(self.process_data(file_path))
                            Y.append(dir)
            self.X, self.Y = X, Y
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type}

    #remove header
    def remove_header(self, lines):
        for i in range(len(lines)):
            if(lines[i] == '\n'):
                start = i+1
                break
        new_lines = lines[start:]
        return new_lines

    def process_data(self, file_path):
        X = []
        with open(file_path, "r", errors ='ignore') as f:
            document = ""
            content = f.readlines()
            content = self.remove_header(content)

            for i in content:
                temp = i.lstrip("> ").replace("/\\","").replace("*","").replace("^","")
                document = document + temp

            X.append(document)
        return X


if __name__ == "__main__":
    import pickle
    file_path = '../../../../data/fednlp/text_classification/20Newsgroups/20news-18828'
    data_loader = DataLoader(file_path)
    train_data_loader = data_loader.data_loader()
    uniform_partition_dict = uniform_partition([train_data_loader["X"], train_data_loader["Y"]])

    # pickle.dump(train_data_loader, open("20news_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("20news_partition.pkl", "wb"))
    print("done")