import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *
import time
class DataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "machine_translation"

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0:
            X, Y = self.process_data(self.data_path[0], self.data_path[1])
            self.X, self.Y = X, Y

        return {"X": self.X, "Y": self.Y, "task_type": self.task_type}

    def process_data(self, source_file_path, target_file_path):
        X = []
        Y = []
        with open(source_file_path, "r") as f:
            for line in f:
                line = line.strip()
                X.append(line)
        with open(target_file_path, "r") as f:
            for line in f:
                line = line.strip()
                Y.append(line)
        return X, Y

if __name__ == "__main__":
    import pickle
    train_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.cs",
                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.en"]
    train_data_loader = DataLoader(train_file_paths)
    train_result = train_data_loader.data_loader()
    uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]])
    # pickle.dump(train_result, open("wmt_cs_en_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_cs_en_partition.pkl", "wb"))
    #
    # train_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.de",
    #                     "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.en"]
    # train_data_loader = DataLoader(train_file_paths)
    # train_result = train_data_loader.data_loader()
    # uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]])
    # pickle.dump(train_result, open("wmt_de_en_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_de_en_partition.pkl", "wb"))
    #
    # train_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.ru",
    #                     "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.en"]
    # train_data_loader = DataLoader(train_file_paths)
    # train_result = train_data_loader.data_loader()
    # uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]])
    # pickle.dump(train_result, open("wmt_ru_en_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_ru_en_partition.pkl", "wb"))
    #
    # train_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.zh",
    #                     "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.en"]
    # train_data_loader = DataLoader(train_file_paths)
    # train_result = train_data_loader.data_loader()
    # uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]])
    # pickle.dump(train_result, open("wmt_zh_en_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_zh_en_partition.pkl", "wb"))
    print("done")
