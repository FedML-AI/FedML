import struct
from tensorflow.core.example import example_pb2
import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.partition import *

# class DataLoader(BaseDataLoader):
#     def __init__(self, data_path, partition, **kwargs):
#         super().__init__(data_path, partition, **kwargs)
#         allowed_keys = {"source_padding", "target_padding", "source_max_sequence_length", "target_max_sequence_length",
#                         "source_vocab_path", "target_vocab_path", "initialize"}
#         self.__dict__.update((key, False) for key in allowed_keys)
#         self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
#
#         X, Y = self.process_data(self.data_path)
#         self.attributes = self.partition(X, Y)
#
#         if self.tokenized:
#             self.source_sequence_length = []
#             self.target_sequence_length = []
#             self.source_vocab = dict()
#             self.target_vocab = dict()
#             if self.source_padding:
#                 self.source_vocab[PAD_TOKEN] = len(self.source_vocab)
#             if self.target_padding:
#                 self.target_vocab[PAD_TOKEN] = len(self.target_vocab)
#             if self.initialize:
#                 self.source_vocab[SOS_TOKEN] = len(self.source_vocab)
#                 self.source_vocab[EOS_TOKEN] = len(self.source_vocab)
#                 self.target_vocab[SOS_TOKEN] = len(self.target_vocab)
#                 self.target_vocab[EOS_TOKEN] = len(self.target_vocab)
#
#     def tokenize_data(self, X, Y):
#         for i in range(len(X)):
#             X[i] = self.tokenize(X[i])
#             Y[i] = self.tokenize(Y[i])
#             self.source_sequence_length.append(len(X[i]))
#             self.target_sequence_length.append(len(Y[i]))
#
#     def data_loader(self, client_idx=None):
#         if client_idx is not None:
#             X, Y = self.process_data(self.data_path, client_idx=client_idx)
#         else:
#             X, Y = self.process_data(self.data_path)
#         if self.tokenized:
#             self.tokenize_data(X, Y)
#         self.X, self.Y = X, Y
#         result = dict()
#
#         if self.tokenized:
#             if self.source_vocab_path:
#                 self.process_vocab(self.source_vocab_path, self.source_vocab)
#             else:
#                 self.build_vocab(self.X, self.source_vocab)
#             result["source_vocab"] = self.source_vocab
#
#             if self.target_vocab_path:
#                 self.process_vocab(self.target_vocab_path, self.target_vocab)
#             else:
#                 self.build_vocab(self.Y, self.target_vocab)
#             result["target_vocab"] = self.target_vocab
#
#             if self.source_padding:
#                 if not self.source_max_sequence_length:
#                     self.source_max_sequence_length = max(self.source_sequence_length)
#                     if self.initialize:
#                         self.source_max_sequence_length += 2
#                 self.padding_data(self.X, self.source_max_sequence_length, self.initialize)
#                 result["source_sequence_length"] = self.source_sequence_length
#                 result["source_max_sequence_length"] = self.source_max_sequence_length
#             if self.target_padding:
#                 if not self.target_max_sequence_length:
#                     self.target_max_sequence_length = max(self.target_sequence_length)
#                     if self.initialize:
#                         self.target_max_sequence_length += 2
#                 self.padding_data(self.Y, self.target_max_sequence_length, self.initialize)
#                 result["target_sequence_length"] = self.target_sequence_length
#                 result["target_max_sequence_length"] = self.target_max_sequence_length
#         result["attributes"] = self.attributes
#         result["X"] = self.X
#         result["Y"] = self.Y
#         return result
#
#     @staticmethod
#     def tokenize(document):
#         tokens = []
#         for token in document.split(" "):
#             token = token.strip()
#             if token:
#                 tokens.append(token)
#         return tokens
#
#     @staticmethod
#     def process_vocab(vocab_path, vocab):
#         with open(vocab_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 token, index = line.split(" ")
#                 if token not in vocab:
#                     vocab[token] = len(vocab)
#
#     def process_data(self, file_path, client_idx=None):
#         file = open(file_path, "rb")
#         X = []
#         Y = []
#         cnt = 0
#         while True:
#             len_bytes = file.read(8)
#             if not len_bytes:
#                 break
#             str_len = struct.unpack('q', len_bytes)[0]
#             example_str = struct.unpack('%ds' % str_len, file.read(str_len))[0]
#             if client_idx is not None and self.attributes["inputs"][cnt] != client_idx:
#                 cnt += 1
#                 continue
#             example = example_pb2.Example.FromString(example_str)
#             article_text = example.features.feature['article'].bytes_list.value[0].decode()
#             abstract_text = example.features.feature['abstract'].bytes_list.value[0].decode()
#             abstract_text = abstract_text.replace("<s>", "").replace("</s>", "")
#
#             X.append(article_text)
#             Y.append(abstract_text)
#             cnt += 1
#         return X, Y
#
#
# def test_performance():
#     import time
#     from pympler import asizeof
#     train_file_path = "../../../../data/fednlp/seq2seq/CNN_Dailymail/finished_files/train.bin"
#     # load all data
#     start = time.time()
#     data_loader = DataLoader(train_file_path, uniform_partition, tokenized=True, source_padding=True,
#                              target_padding=True)
#     train_data_loader = data_loader.data_loader()
#     end = time.time()
#     print("all data(tokenized):", end - start)
#     print("size", len(train_data_loader["X"]))
#     print("memory cost", asizeof.asizeof(train_data_loader))
#     # load a part of data
#     start = time.time()
#     data_loader = DataLoader(train_file_path, uniform_partition, tokenized=True, source_padding=True,
#                              target_padding=True)
#     train_data_loader = data_loader.data_loader(0)
#     end = time.time()
#     print("part of data(tokenized):", end - start)
#     print("size", len(train_data_loader["X"]))
#     print("memory cost", asizeof.asizeof(train_data_loader))
#
#     # load all data
#     start = time.time()
#     data_loader = DataLoader(train_file_path, uniform_partition)
#     train_data_loader = data_loader.data_loader()
#     end = time.time()
#     print("all data:", end - start)
#     print("size", len(train_data_loader["X"]))
#     print("memory cost", asizeof.asizeof(train_data_loader))
#     # load a part of data
#     start = time.time()
#     data_loader = DataLoader(train_file_path, uniform_partition)
#     train_data_loader = data_loader.data_loader(0)
#     end = time.time()
#     print("part of data:", end - start)
#     print("size", len(train_data_loader["X"]))
#     print("memory cost", asizeof.asizeof(train_data_loader))


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
