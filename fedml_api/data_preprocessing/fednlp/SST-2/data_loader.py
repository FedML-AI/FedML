import os
import math
import random
from random import shuffle
import sys
import csv
import time


sys.path.append('..')

from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *

# if download with script in data folder 
# data_dir shoule be '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank'

class DataLoader(BaseDataLoader):
    def __init__(self, data_path, sentence_index, label_file, partition, **kwargs):
        super().__init__(data_path, partition, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "tokenized", "source_max_sequence_length",
                        "target_max_sequence_length", "vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.source_sequence_length = []
        self.target_sequence_length = []
        self.sentence_index = sentence_index
        self.label_file = label_file
        self.attributes = dict()
        self.attributes['inputs'] = []
        self.label_vocab = {1:"very",2:"negative",3:"neutral",4:"positive"}
        self.label_dict = dict()


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
    
    def label_level(self,label):
        label = float(label)
        if label >= 0.0 and label <= 0.2:
            return "very negative"
        elif label > 0.2 and label <= 0.4:
            return "negative"
        elif label > 0.4 and label <= 0.6:
            return "neutral"
        elif label > 0.6 and label <= 0.8:
            return "positive"
        else:
            return "very positive"

    def process_data(self,client_idx=None):
        cnt = 0
        max_source_length = -1
        
        with open(self.label_file) as f2:
            for label_line in f2:
                label = label_line.split('|')
                self.label_dict[label[0].strip()] = label[1]
                

        for i in self.sentence_index:
            if client_idx is not None and client_idx != self.attributes["inputs"][cnt]:
                cnt+=1
                continue
            data = i.split('|')

            if self.tokenized:
                tokens = self.tokenize(data[0].strip())
                self.X.append(tokens)
            else:
                tokens = data[0].strip()
                self.X.append([tokens])
                
            self.source_sequence_length.append(len(tokens))
            max_source_length = max(len(tokens),max_source_length)
            self.target_sequence_length.append(1)   
            self.Y.append([self.label_level(self.label_dict[data[1].strip()])])


        return max_source_length, 1


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
            self.padding_data(self.X, max_source_length, self.initialize)

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
    data_path = '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank/'
    data_file_path = '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank/dictionary.txt'
    label_file_path = '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank/sentiment_labels.txt'

    train_indexes = []
    test_indexes = []
    with open(data_file_path,"r",encoding="utf-8") as f:
        files = f.readlines()
        shuffle(files)
        train_indexes = files[0:int(len(files)*0.8)]
        test_indexes = files[int(len(files)*0.8):]

    train_data_loader = DataLoader(data_file_path, train_indexes, label_file_path, uniform_partition)

    test_data_loader = DataLoader(data_file_path, test_indexes, label_file_path, uniform_partition)


    result = train_data_loader.data_loader()
    print(len(result['X']))
    print(len(result['attributes']['inputs']))
    print(result['X'][0:10])
    print(result['Y'][0:10])
    print(result['source_sequence_length'][140:150])
    print(result['source_max_sequence_length'])
