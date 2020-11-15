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

# if download with script in data folder 
# data_dir shoule be '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data'

# class DataLoader(BaseDataLoader):
#     def __init__(self, data_path, partition, **kwargs):
#         super().__init__(data_path,partition, **kwargs)
#         allowed_keys = {"source_padding", "target_padding", "tokenized", "source_max_sequence_length",
#                         "target_max_sequence_length", "vocab_path", "initialize"}
#         self.__dict__.update((key, False) for key in allowed_keys)
#         self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
#         self.source_sequence_length = []
#         self.target_sequence_length = []
#         self.title = []
#         self.attributes = dict()
#         self.attributes['inputs'] = []
#         self.attr_label = dict()
#         self.label_vocab = dict()
#
#
#
#         if self.tokenized:
#             self.vocab = dict()
#             if self.initialize:
#                 self.vocab[SOS_TOKEN] = len(self.vocab)
#                 self.vocab[EOS_TOKEN] = len(self.vocab)
#         if self.source_padding or self.target_padding:
#             self.vocab[PAD_TOKEN] = len(self.vocab)
#             self.label_vocab[PAD_TOKEN] = len(self.vocab)
#
#
#     def tokenize(self,document,switch):
#         # Create a blank Tokenizer with just the English vocab
#         tokens = [str(token) for token in spacy_tokenizer.en_tokenizer(document)]
#         if switch == 'data':
#             for i in list(tokens):
#                 if i not in self.vocab:
#                     self.vocab[i] = len(self.vocab)
#         else:
#             for i in list(tokens):
#                 if i not in self.label_vocab:
#                     self.label_vocab[i] = len(self.label_vocab)
#         return tokens
#
#     def process_attributes(self):
#         length = len(set(self.attributes['inputs']))
#         self.attributes['n_clients'] = length
#         return self.attributes
#
#     def process_data(self,client_idx=None):
#         with open(self.data_path, "r", encoding='utf-8') as f:
#             cnt = 0
#             data  = f.readlines()
#             max_source_length = -1
#             max_target_length = -1
#             for i in range(len(data)):
#                 if client_idx is not None and client_idx != self.attributes["inputs"][cnt]:
#                     cnt+=1
#                     continue
#                 if len(data[i]) > 1 and data[i][0].isdigit():
#                     clean_data = data[i].split('\t')[1].strip().replace('"',"")
#                     if self.tokenized:
#                         tokens = self.tokenize(clean_data,'data')
#                         self.X.append(tokens)
#                     else:
#                         tokens = clean_data
#                         self.X.append([tokens])
#                     self.source_sequence_length.append(len(tokens))
#                     max_source_length = max(max_source_length,len(tokens))
#
#                 elif len(data[i-1]) > 1 and data[i-1][0].isdigit():
#                     label = data[i].rstrip("\n")
#                     if label not in self.attr_label:
#                         self.attr_label[label] = len(self.attr_label)
#                     self.attributes['inputs'].append(self.attr_label[label])
#                     tokens = self.tokenize(label,'label')
#                     self.target_sequence_length.append(len(tokens))
#                     self.Y.append(tokens)
#                     max_target_length = max(max_target_length,len(tokens))
#
#
#             return max_source_length, max_target_length
#
#
#     def data_loader(self,client_idx=None):
#         result = dict()
#         if client_idx is not None:
#             max_source_length , max_target_length = self.process_data(client_idx)
#         else:
#             max_source_length , max_target_length = self.process_data()
#         if self.source_padding:
#             self.padding_data(self.X, max_source_length,self.initialize)
#         if self.target_padding:
#             self.padding_data(self.Y, max_target_length,self.initialize)
#
#         if callable(self.partition):
#             self.attributes = self.partition(self.X, self.Y)
#         else:
#             self.attributes = self.process_attributes()
#
#         if self.tokenized:
#                 result['vocab'] = self.vocab
#
#         result['X'] = self.X
#         result['Y'] = self.Y
#         result['label_vocab'] = self.label_vocab
#         result['attributes'] = self.attributes
#         result['source_sequence_length'] = self.source_sequence_length
#         result['target_sequence_length'] = self.target_sequence_length
#         result['source_max_sequence_length'] = max_source_length
#         result['target_max_sequence_length'] = max_target_length
#
#         return result


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
    data_path = '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data'
    train_file_path = '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.txt'
    test_file_path = '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt'
    train_data_loader = DataLoader(train_file_path)
    train_result = train_data_loader.data_loader()

    test_data_loader = DataLoader(test_file_path)
    test_result = test_data_loader.data_loader()
    uniform_partition_dict = uniform_partition([train_result["X"], train_result["Y"]],
                                               [test_result["X"], test_result["Y"]])
    print("done")


