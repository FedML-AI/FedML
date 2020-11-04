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
            self.label_vocab = dict()
            if self.initialize:
                self.vocab[SOS_TOKEN] = len(self.vocab)
                self.vocab[EOS_TOKEN] = len(self.vocab)            
        if self.source_padding or self.target_padding:
            self.vocab[PAD_TOKEN] = len(self.vocab)
            self.label_vocab[PAD_TOKEN] = len(self.vocab)


    def tokenize(self,document,switch):
        # Create a blank Tokenizer with just the English vocab
        tokens = [str(token) for token in spacy_tokenizer.en_tokenizer(document)]
        if switch == 'data':
            for i in list(tokens):
                if i not in self.vocab:
                    self.vocab[i] = len(self.vocab)
        else:
            for i in list(tokens):
                if i not in self.label_vocab:
                    self.label_vocab[i] = len(self.label_vocab)
        return tokens

    def process_data(self):
        with open(self.data_path, "r", encoding='utf-8') as f:
            data  = f.readlines()
            max_source_length = -1
            max_target_length = -1
            for i in range(len(data)):
                if len(data[i]) > 1 and data[i][0].isdigit():
                    clean_data = data[i].split('\t')[1]
                    tokens = self.tokenize(clean_data,'data')
                    self.source_sequence_length.append(len(tokens))
                    self.X.append(tokens)
                    max_source_length = max(max_source_length,len(tokens))

                elif len(data[i-1]) > 1 and data[i-1][0].isdigit():
                    label = data[i].rstrip("\n")
                    tokens = self.tokenize(label,'label')
                    self.target_sequence_length.append(len(tokens))
                    self.Y.append(tokens)
                    max_target_length = max(max_target_length,len(tokens))


            return max_source_length, max_target_length
    

    def data_loader(self):
        result = dict()
        max_source_length, max_target_length = self.process_data()
        self.padding_data(self.X, max_source_length,self.initialize)
        self.padding_data(self.Y, max_target_length,self.initialize)

        result['X'] = self.X
        result['Y'] = self.Y
        result['vocab'] = self.vocab
        result['label_vocab'] = self.label_vocab
        result['source_sequence_length'] = self.source_sequence_length
        result['target_sequence_length'] = self.target_sequence_length
        result['max_source_length'] = max_source_length

        return result


if __name__ == "__main__":
    data_path = '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data'
    train_file_path = '../../../../data//fednlp/text_classification/SemEval2010Task8/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.txt'
    data_loader = DataLoader(train_file_path, tokenized=True, source_padding=True, target_padding=True)

    result = data_loader.data_loader()
