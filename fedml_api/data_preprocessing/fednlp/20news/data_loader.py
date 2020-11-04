import os
import math
import random
import sys
import time


sys.path.append('..')

from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *

# if download with script in data folder 
# data_dir shoule be '../../../../data/fednlp/text_classification/20Newsgroups/20news-18828'

class DataLoader(BaseDataLoader):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "source_max_sequence_length",
                        "target_max_sequence_length", "vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.source_sequence_length = []
        self.target_sequence_length = []


        if self.tokenized:
            self.vocab = dict()
            self.label_vocab = dict()
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
    

    #parse all the data set 
    def process_data(self,files):
        document = ""
        file_path = files[0]
        with open(file_path,"r",errors = 'ignore') as f:
            content = f.readlines()
            content = self.remove_header(content)
            for i in content:
                temp = i.lstrip("> ").replace("/\\","").replace("*","").replace("^","")
                document = document + temp   
            return files[1], document

    def data_loader(self):
        max_source_length = -1
        max_target_length = -1
        document = []
        result = dict()

        for j in self.data_path:
            datas = self.process_data(j)
            for label, document in datas:
                tokens = self.tokenize(document) 
                labels = label.split('.')
                for i in labels:
                    if i not in self.label_vocab:
                        self.label_vocab[i] = len(self.label_vocab)
                self.Y.append([label])
                self.X.append(tokens)
                self.source_sequence_length.append(len(tokens))
                max_source_length = max(len(tokens),max_source_length)
                self.target_sequence_length.append(len(labels))
                max_target_length = max(len(labels),max_target_length)

        self.padding_data(self.X, max_source_length,self.initialize)

        self.padding_data(self.Y,max_target_length,self.initialize)

        
        
        result['X'] = self.X
        result['Y'] = self.Y
        result['vocab'] = self.vocab
        result['label_vocab'] = self.label_vocab
        result['source_sequence_length'] = self.source_sequence_length
        result['target_sequence_length'] = self.target_sequence_length
        result['max_source_length'] = max_source_length

        return result

if __name__ == "__main__":
    file_path = '../../../../data/fednlp/text_classification/20Newsgroups/20news-18828'
    folders = [f for f in os.listdir(file_path)]
    files = []
    for folder_name in folders:
        folder_path = os.path.join(file_path, folder_name)
        files.extend([(os.path.join(folder_path,f), folder_name) for f in os.listdir(folder_path)])
    random.seed(3)
    random.shuffle(files)
    train_files_path = files[0:int(len(files)*0.8)]
    test_files_path = files[int(len(files)*0.8):]
    data_loader = DataLoader(train_files_path, tokenized=True, source_padding=True, target_padding=True)
    train_data_loader = data_loader.data_loader()
    #print(train_data_loader['Y'])
    print("done")