import os
import math
import random
import sys
import csv
import time
import datetime


sys.path.append('..')

from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *

# data_dir shoule be '../../../../data//fednlp/text_classification/Sentiment140/'

class DataLoader(BaseDataLoader):
    def __init__(self, data_path, partition, **kwargs):
        super().__init__(data_path, partition, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "tokenized", "source_max_sequence_length",
                        "target_max_sequence_length", "vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.source_sequence_length = []
        self.target_sequence_length = []
        self.title = []
        self.attributes = dict()
        self.attributes['inputs'] = []
        self.label_vocab = {'2':'neutral','0':'negative','4':'positive'}


        if self.tokenized:
            self.vocab = dict()
            if self.initialize:
                self.vocab[SOS_TOKEN] = len(self.vocab)
                self.vocab[EOS_TOKEN] = len(self.vocab)            
        if self.source_padding or self.target_padding:
            self.vocab[PAD_TOKEN] = len(self.vocab)
            self.label_vocab[PAD_TOKEN] = len(self.vocab)

    def tokenize(self,document):
        # Create a blank Tokenizer with just the English vocab
        tokens = [str(token) for token in spacy_tokenizer.en_tokenizer(document)]
        for i in list(tokens):
            if i not in self.vocab:
                self.vocab[i] = len(self.vocab)
        return tokens

    def process_data(self,client_idx=None):
        with open(self.data_path ,"r",newline='',encoding='utf-8',errors='ignore') as csvfile:
            cnt = 0
            data = csv.reader(csvfile,delimiter=',')
            max_source_length = -1
            for line in data:
                if client_idx is not None and client_idx != self.attributes["inputs"][cnt]:
                    cnt+=1
                    continue
                if self.tokenized:
                    tokens = self.tokenize(line[5])
                    self.X.append(tokens)
                else:
                    tokens = line[5]
                    self.X.append([tokens])
                self.source_sequence_length.append(len(tokens))
                max_source_length = max(max_source_length,len(tokens))
                self.Y.append(self.label_vocab[line[0]])
                self.target_sequence_length.append(1)

        return max_source_length, 1

    def process_attributes(self):
        length = len(set(self.attributes['inputs']))
        self.attributes['n_clients'] = length
        return self.attributes
    
    def data_loader(self,client_idx=None):
        result = dict()
        if client_idx is not None:
            max_source_length , max_target_length = self.process_data(client_idx)
        else:
            max_source_length , max_target_length = self.process_data()

        if callable(self.partition):
            self.attributes = self.partition(self.X, self.Y)
        else:
            self.attributes = self.process_attributes()

        if self.source_padding:
            self.padding_data(self.X, max_source_length,self.initialize)
        
        if self.tokenized:
            result['vocab'] = self.vocab

        result['X'] = self.X
        result['Y'] = self.Y
        result['label_vocab'] = self.label_vocab
        result['attributes'] = self.attributes
        result['source_sequence_length'] = self.source_sequence_length
        result['target_sequence_length'] = self.target_sequence_length
        result['source_max_sequence_length'] = max_source_length
        result['target_max_sequence_length'] = max_target_length


        return result





if __name__ == "__main__":
    data_path = '../../../../data/fednlp/text_classification/Sentiment140/'
    test_file_path = '../../../../data/fednlp/text_classification/Sentiment140/testdata.manual.2009.06.14.csv'
    train_file_path = '../../../../data/fednlp/text_classification/Sentiment140/training.1600000.processed.noemoticon.csv'
    test_data_loader = DataLoader(test_file_path, uniform_partition)

    train_data_loader = DataLoader(train_file_path, uniform_partition)

    result = train_data_loader.data_loader()
    print(len(result['X']))
    print(len(result['attributes']['inputs']))
    print(result['source_max_sequence_length'])
    print(result['X'][140:150])
    print(result['Y'][140:150])

    print(result['source_sequence_length'][140:150])
