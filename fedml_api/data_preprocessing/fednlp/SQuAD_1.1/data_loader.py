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
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
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

    def process_data(self):
        with open(self.data_path,"r",encoding='utf-8') as f:
            data = json.load(f)
            max_document_length = -math.inf
            max_question_length = -math.inf

            for document in data["data"]:
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


    def data_loader(self):
        result = dict()

        max_document_length, max_question_length = self.process_data()
        self.padding_data(self.document_X, max_document_length,self.initialize)
        self.padding_data(self.question_X, max_question_length,self.initialize)

        result['document_X'] = self.document_X
        result['question_X'] = self.question_X
        result['Y'] = self.Y
        result['question_id'] = self.question_id
        result['vocab'] = self.vocab
        result['qas_vocab'] = self.qas_vocab
        result['source_sequence_length'] = self.source_sequence_length
        result['question_sequence_length'] = self.question_sequence_length
        result['max_document_length'] = max_document_length
        result['max_question_length'] = max_question_length

        return result

if __name__ == "__main__":
    data_path = '../../../../data/fednlp/span_extraction/SQuAD_1.1'
    train_file_path = '../../../../data//fednlp/span_extraction/SQuAD_1.1/train-v1.1.json'

    data_loader = DataLoader(train_file_path, tokenized=True, source_padding=True, target_padding=True)

    result = data_loader.data_loader()
    print(result['Y'][0:20])
