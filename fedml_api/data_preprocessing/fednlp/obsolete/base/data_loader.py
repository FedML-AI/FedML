from abc import ABC, abstractmethod
from .globals import *


# class BaseDataLoader(ABC):
#     @abstractmethod
#     def __init__(self, data_path, partition, **kwargs):
#         allowed_keys = {"tokenized"}
#         self.__dict__.update((key, False) for key in allowed_keys)
#         self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
#         self.data_path = data_path
#         self.partition = partition
#         self.X = []
#         self.Y = []
#
#     @abstractmethod
#     def data_loader(self, client_idx=None):
#         pass
#
#     @abstractmethod
#     def process_data(self, file_path):
#         pass
#
#     @staticmethod
#     def padding_data(x, max_sequence_length, initialize, pad_token=PAD_TOKEN):
#         max_sequence_length = max_sequence_length-2 if initialize else max_sequence_length
#         for i, single_x in enumerate(x):
#             if len(single_x) <= max_sequence_length:
#                 if initialize:
#                     single_x = [SOS_TOKEN] + single_x + [EOS_TOKEN]
#                 for _ in range(len(single_x), max_sequence_length):
#                     single_x.append(pad_token)
#             else:
#                 single_x = single_x[:max_sequence_length]
#                 if initialize:
#                     single_x = [SOS_TOKEN] + single_x + [EOS_TOKEN]
#
#     @staticmethod
#     def raw_data_to_idx(x, token_vocab):
#         idx_x = []
#         for i, single_x in enumerate(x):
#             idx_single_x = []
#             for j, token in enumerate(single_x):
#                 idx_single_x.append(token_vocab[token] if token in token_vocab else token_vocab[UNK_TOKEN])
#             idx_x.append(idx_single_x)
#         return idx_x
#
#     @staticmethod
#     def build_vocab(x, vocab):
#         for single_x in x:
#             for token in single_x:
#                 if token not in vocab:
#                     vocab[token] = len(vocab)

class BaseDataLoader(ABC):
    @abstractmethod
    def __init__(self, data_path):
        self.data_path = data_path
        self.X = []
        self.Y = []

    @abstractmethod
    def data_loader(self, client_idx=None):
        pass

    @abstractmethod
    def process_data(self, file_path):
        pass
