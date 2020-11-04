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

# data_dir shoule be '../../../../data//fednlp/text_classification/Sentiment140/'

class DataLoader(BaseDataLoader):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "source_max_sequence_length",
                        "target_max_sequence_length", "vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.source_sequence_length = []
        self.target_sequence_length = []
        self.title = []

        if self.tokenized:
            self.vocab = dict()
            self.label_vocab = {'2':'neutral','0':'negative','4':'positive'}
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

    def process_data(self):
        with open(self.data_path ,"r",newline='',encoding='utf-8',errors='ignore') as csvfile:
            data = csv.reader(csvfile,delimiter=',')
            max_source_length = -1
            for line in data:
                tokens = self.tokenize(line[5])
                self.X.append(tokens)
                self.source_sequence_length.append(len(tokens))
                max_source_length = max(max_source_length,len(tokens))
                self.Y.append(self.label_vocab[line[0]])
                self.target_sequence_length.append(1)

        return max_source_length, 1


    def data_loader(self):
        result = dict()
        max_source_length, max_target_length = self.process_data()
        print(max_source_length)
        self.padding_data(self.X, max_source_length,self.initialize)

        result['X'] = self.X
        result['Y'] = self.Y
        result['vocab'] = self.vocab
        result['label_vocab'] = self.label_vocab
        result['source_sequence_length'] = self.source_sequence_length
        result['target_sequence_length'] = self.target_sequence_length
        result['max_source_length'] = max_source_length

        return result





if __name__ == "__main__":
    data_path = '../../../../data//fednlp/text_classification/Sentiment140/'
    train_file_path = '../../../../data//fednlp/text_classification/Sentiment140/training.1600000.processed.noemoticon.csv'

    data_loader = DataLoader(train_file_path, tokenized=True, source_padding=True, target_padding=True)

    result = data_loader.data_loader()
    print(result['X'][0:10])

