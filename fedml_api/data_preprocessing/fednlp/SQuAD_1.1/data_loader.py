import os
import math
import random
import sys
import json
import time


sys.path.append('..')

from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *


# data_dir shoule be '../../../../data/fednlp/span_extraction/SQuAD_1.1'


class DataLoader(BaseDataLoader):
    def __init__(self, data_path, partition, **kwargs):
        super().__init__(data_path, partition, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "source_max_sequence_length",
                        "target_max_sequence_length", "vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.source_sequence_length = []
        self.question_sequence_length = []
        self.document_X = []
        self.question_X = []
        self.question_id = []
        self.answer_X = []
        self.attributes = dict()
        self.attributes['inputs'] = []

        if self.tokenized:
            self.vocab = dict()
            self.qas_vocab = dict()
            if self.initialize:
                self.vocab[SOS_TOKEN] = len(self.vocab)
                self.vocab[EOS_TOKEN] = len(self.vocab)            
        if self.source_padding or self.target_padding:
            self.vocab[PAD_TOKEN] = len(self.vocab)
            self.qas_vocab[PAD_TOKEN] = len(self.vocab)
    def tokenize(self,document):
        # Create a blank Tokenizer with just the English vocab
        tokens = [str(token) for token in spacy_tokenizer.en_tokenizer(document)]
        return tokens
    def process_attributes(self):
        self.attributes['n_clients'] = len(self.attributes['inputs'])
        print(self.attributes['n_clients'])

    def process_data(self,client_idx=None):
        with open(self.data_path,"r",encoding='utf-8') as f:
            data = json.load(f)
            max_document_length = -math.inf
            max_question_length = -math.inf

            for index, document in enumerate(data["data"]):
                if client_idx is not None and client_idx != self.attributes["inputs"][cnt]:
                    cnt+=1
                    continue
                self.attributes['inputs'].append(index)
                single_document = "".join([paragraph["context"] for paragraph in document["paragraphs"]])
                document_tokens = self.tokenize(single_document)
                self.document_X.append(document_tokens)
                self.source_sequence_length.append(len(document_tokens))
                max_document_length = max(max_document_length,len(document_tokens))
                for i in document_tokens:
                    if i not in self.vocab:
                        self.vocab[i] = len(self.vocab)

                for paragraph in document["paragraphs"]:

                    for qas in paragraph["qas"]:
                        question = qas["question"]
                        question_tokens = self.tokenize(question)
                        max_question_length = max(max_question_length,len(question_tokens))
                        for i in question_tokens:
                            if i not in self.qas_vocab:
                                self.qas_vocab[i] = len(self.qas_vocab)

                        answer  = []
                        for answers in qas["answers"]:
                            if(answers["text"] not in answer):
                                answer.append(answers["text"])
                                self.question_X.append(question_tokens)
                                self.question_sequence_length.append(len(question_tokens))
                                self.question_id.append(qas["id"])
                                start = answers["answer_start"]
                                end = start + len(answers["text"].rstrip())
                                self.Y.append([start,end])
        return max_document_length, max_question_length


    def data_loader(self,client_idx=None):
        result = dict()

        if client_idx is not None:
            max_document_length , max_question_length = self.process_data(client_idx)
        else:
            max_document_length , max_question_length = self.process_data()

        if self.source_padding:
            self.padding_data(self.document_X, max_document_length,self.initialize)
        if self.target_padding:
            self.padding_data(self.question_X, max_question_length,self.initialize)

        if callable(self.partition):
            self.attributes = self.partition(self.document_X, self.Y)
        else:
            self.process_attributes()

        result['document_X'] = self.document_X
        result['question_X'] = self.question_X
        result['Y'] = self.Y
        result['question_id'] = self.question_id
        result['vocab'] = self.vocab
        result['question_vocab'] = self.qas_vocab
        result['attributes'] = self.attributes
        result['source_sequence_length'] = self.source_sequence_length
        result['question_sequence_length'] = self.question_sequence_length
        result['document_max_sequence_length'] = max_document_length
        result['question_max_sequence_length'] = max_question_length

        return result

if __name__ == "__main__":
    data_path = '../../../../data/fednlp/span_extraction/SQuAD_1.1'
    train_file_path = '../../../../data//fednlp/span_extraction/SQuAD_1.1/train-v1.1.json'

    data_loader = DataLoader(train_file_path, uniform_partition, tokenized=True, source_padding=True, target_padding=True)

    result = data_loader.data_loader()
    print(len(result['document_X']))
    print(len(result['attributes']['inputs']))
    print(result['attributes']['inputs'])