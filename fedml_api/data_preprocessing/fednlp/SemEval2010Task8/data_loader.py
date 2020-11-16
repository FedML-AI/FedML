import os
import math
import random
import sys
import csv
import time


sys.path.append('..')

from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *

class DataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "classification"
        self.target_vocab = None

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
            X, Y = self.process_data(self.data_path)
            self.X, self.Y = X, Y
            self.target_vocab = {key: i for i, key in enumerate(set(Y))}
        return {"X": self.X, "Y": self.Y, "target_vocab": self.target_vocab, "task_type": self.task_type}

    def process_data(self, file_path):
        X = []
        Y = []
        with open(file_path, "r", encoding='utf-8') as f:
            data = f.readlines()
            for i in range(len(data)):
                if len(data[i]) > 1 and data[i][0].isdigit():
                    clean_data = data[i].split('\t')[1].strip().replace('"',"")
                    X.append(clean_data)

                elif len(data[i-1]) > 1 and data[i-1][0].isdigit():
                    label = data[i].rstrip("\n")
                    Y.append(label)
        return X, Y


if __name__ == "__main__":
    import pickle
    data_path = '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data'
    train_file_path = '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.txt'
    test_file_path = '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt'
    train_data_loader = DataLoader(train_file_path)
    train_result = train_data_loader.data_loader()

    test_data_loader = DataLoader(test_file_path)
    test_result = test_data_loader.data_loader()
    uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]],
                                               [test_result["X"], test_result["Y"]])
    # pickle_dict = train_result
    # pickle_dict["X"].extend(test_result["X"])
    # pickle_dict["Y"].extend(test_result["Y"])
    # pickle.dump(pickle_dict, open("semeval_2010_task8_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("semeval_2010_task8_partition.pkl", "wb"))
    print("done")


