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
# data_dir shoule be '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank'

class DataLoader(BaseDataLoader):
    def __init__(self, data_path, sentence_index, label_file, partition, **kwargs):
        super().__init__(data_path, partition, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "source_max_sequence_length",
                        "target_max_sequence_length", "vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.source_sequence_length = []
        self.target_sequence_length = []
        self.sentence_index = sentence_index
        self.label_file = label_file
        self.attributes = dict()
        self.attributes['inputs'] = []

        if self.tokenized:
            self.vocab = dict()
            self.label_vocab = {1:"very",2:"negative",3:"neutral",4:"positive"}
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
        with open(self.data_path,"r", encoding='utf-8') as f1 , open(self.label_file) as f2:
            max_source_length = -1
            for data_line , label_line in zip(f1,f2):
                if client_idx is not None and client_idx != self.attributes["inputs"][cnt]:
                    cnt+=1
                    continue
                data = data_line.split('\t')
                label = label_line.split('|')
                if data[0] in self.sentence_index:
                    tokens = self.tokenize(data[1].strip())
                    self.X.append(tokens)
                    self.source_sequence_length.append(len(tokens))
                    max_source_length = max(len(tokens),max_source_length)

                if label[0] in self.sentence_index:
                    self.target_sequence_length.append(1)

                    self.Y.append([self.label_level(label[1])])
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

        result['X'] = self.X
        result['Y'] = self.Y
        result['vocab'] = self.vocab
        result['label_vocab'] = self.label_vocab
        result['attributes'] = self.attributes
        result['source_sequence_length'] = self.source_sequence_length
        result['target_sequence_length'] = self.target_sequence_length
        result['source_max_sequence_length'] = max_source_length
        result['target_max_sequence_length'] = max_target_length

        return result


if __name__ == "__main__":
    data_path = '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank/'
    data_split_file = '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank/datasetSplit.txt'
    data_file_path = '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank/datasetSentences.txt'
    label_file_path = '../../../../data//fednlp/text_classification/SST-2/stanfordSentimentTreebank/sentiment_labels.txt'

    train_indexes = []
    test_indexes = []
    with open(data_split_file,"r",encoding="utf-8") as f:
        for line in f:
            data = line.split(',')
            if data[1].strip() == '1':
                train_indexes.append(data[0])
            elif data[1].strip() == '2':
                test_indexes.append(data[0]) 
            else:
                continue

    data_loader = DataLoader(data_file_path, train_indexes, label_file_path, uniform_partition, \
                                tokenized=True, source_padding=True, target_padding=True)

    result = data_loader.data_loader()
    print(len(result['X']))
    print(len(result['attributes']['inputs']))
