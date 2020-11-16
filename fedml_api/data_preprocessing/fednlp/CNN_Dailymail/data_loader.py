import struct
from tensorflow.core.example import example_pb2
import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.partition import *

class DataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "summarization"

    def data_loader(self):
        if len(self.X) == 0 or len(self.Y) == 0:
            X, Y = self.process_data(self.data_path)
            self.X, self.Y = X, Y

        return {"X": self.X, "Y": self.Y, "task_type": self.task_type}

    def process_data(self, file_path):
        file = open(file_path, "rb")
        X = []
        Y = []
        while True:
            len_bytes = file.read(8)
            if not len_bytes:
                break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, file.read(str_len))[0]
            example = example_pb2.Example.FromString(example_str)
            article_text = example.features.feature['article'].bytes_list.value[0].decode().strip()
            abstract_text = example.features.feature['abstract'].bytes_list.value[0].decode()
            abstract_text = abstract_text.replace("<s>", "").replace("</s>", "").strip()

            X.append(article_text)
            Y.append(abstract_text)
        return X, Y

if __name__ == "__main__":
    import pickle
    train_file_path = "../../../../data/fednlp/seq2seq/CNN_Dailymail/finished_files/train.bin"
    dev_file_path = "../../../../data/fednlp/seq2seq/CNN_Dailymail/finished_files/val.bin"
    test_file_path = "../../../../data/fednlp/seq2seq/CNN_Dailymail/finished_files/test.bin"
    vocab_file_path = "../../../../data/fednlp/seq2seq/CNN_Dailymail/finished_files/vocab"
    train_data_loader = DataLoader(train_file_path)
    train_result = train_data_loader.data_loader()

    test_data_loader = DataLoader(test_file_path)
    test_result = test_data_loader.data_loader()

    uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]],
                                               [test_result["X"], test_result["Y"]])

    # pickle_dict = train_result
    # pickle_dict["X"].extend(test_result["X"])
    # pickle_dict["Y"].extend(test_result["Y"])
    # pickle.dump(pickle_dict, open("cnn_daily_mail_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict}, open("cnn_daily_mail_partition.pkl", "wb"))
    print("done")
