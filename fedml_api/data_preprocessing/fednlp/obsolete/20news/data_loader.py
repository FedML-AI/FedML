import os
import math
import random
import sys
import time
sys.path.append('..')

from base.data_loader import BaseDataLoader
from base.utils import *
from base.partition import *

if download with script in data folder 
data_dir shoule be '../../../../data/fednlp/text_classification/20Newsgroups/20news-18828'

class DataLoader(BaseDataLoader):
    def __init__(self, data_path, partition, **kwargs):
        super().__init__(data_path, partition, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "tokenized" "source_max_sequence_length",
                        "target_max_sequence_length", "vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.source_sequence_length = []
        self.target_sequence_length = []
        self.attributes = dict()
        self.attributes['inputs'] = []
        self.label_vocab = dict()

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



    #remove header
    def remove_header(self,lines):
        for i in range(len(lines)):
            if(lines[i] == '\n'):
                start = i+1
                break
        new_lines = lines[start:]
        return new_lines
    def process_attributes(self):
        self.attributes["n_clients"] = len(self.data_path)
        return self.attributes


    #parse all the data set
    def process_data(self,file_path,client_idx=None):
        cnt = 0
        for index, files in enumerate(file_path):
            document = ""
            file_path = files[0]
            with open(file_path,"r",errors = 'ignore') as f:
                content = f.readlines()
                content = self.remove_header(content)
                if client_idx is not None and client_idx != self.attributes["inputs"][cnt]:
                    cnt+=1
                    continue
                for i in content:
                    temp = i.lstrip("> ").replace("/\\","").replace("*","").replace("^","")
                    document = document + temp
                label = files[1]
                if self.tokenized:
                    tokens = self.tokenize(document)
                    self.X.append(tokens)
                else:
                    tokens = document
                    self.X.append([tokens])
                labels = label.split('.')
                for i in labels:
                    if i not in self.label_vocab:
                        self.label_vocab[i] = len(self.label_vocab)
                self.Y.append([label])

                self.attributes['inputs'].append(index)
                self.source_sequence_length.append(len(tokens))
                self.target_sequence_length.append(len(labels))


        return len(tokens), len(labels)

    def data_loader(self,client_idx=None):
        max_source_length = -1
        max_target_length = -1
        document = []
        result = dict()
        if client_idx is not None:
            source_length, target_length  = self.process_data(self.data_path, client_idx)
        else:
            source_length, target_length  = self.process_data(self.data_path)

        max_source_length = max(source_length, max_source_length)
        max_target_length = max(target_length, max_target_length)

        if callable(self.partition):
            self.attributes = self.partition(self.X, self.Y)
        else:
            self.attributes = self.process_attributes()

        if self.source_padding:
            self.padding_data(self.X, max_source_length,self.initialize)
        if self.target_padding:
            self.padding_data(self.Y,max_target_length,self.initialize)

        if self.tokenized:
            result['vocab'] = self.vocab



        result['X'] = self.X
        result['Y'] = self.Y
        result['attributes'] = self.attributes
        result['label_vocab'] = self.label_vocab
        result['source_sequence_length'] = self.source_sequence_length
        result['target_sequence_length'] = self.target_sequence_length
        result['source_max_sequence_length'] = max_source_length
        result['target_max_sequence_length'] = max_target_length

        return result


