import json
from abc import ABC, abstractmethod

import h5py
import numpy as np
from tqdm import tqdm


class BaseRawDataLoader(ABC):
    @abstractmethod
    def __init__(self, data_path):
        self.data_path = data_path
        self.attributes = dict()
        self.attributes["index_list"] = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def process_data_file(self, file_path):
        pass

    @abstractmethod
    def generate_h5_file(self, file_path):
        pass


class TextClassificationRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super(TextClassificationRawDataLoader, self).__init__(data_path)
        self.X = dict()
        self.Y = dict()
        self.attributes["num_labels"] = -1
        self.attributes["label_vocab"] = None
        self.attributes["task_type"] = "text_classification"

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.X.keys():
            f["X/" + str(key)] = self.X[key]
            f["Y/" + str(key)] = self.Y[key]
        f.close()


class SpanExtractionRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super(SpanExtractionRawDataLoader, self).__init__(data_path)
        self.context_X = dict()
        self.question_X = dict()
        self.Y = dict()
        self.attributes["task_type"] = "span_extraction"
        self.Y_answer = dict()

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.context_X.keys():
            f["context_X/" + str(key)] = self.context_X[key]
            f["question_X/" + str(key)] = self.question_X[key]
            f["Y/" + str(key)] = self.Y[key]
            f["Y_answer/" + str(key)] = self.Y_answer[key]
        f.close()


class SeqTaggingRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super(SeqTaggingRawDataLoader, self).__init__(data_path)
        self.X = dict()
        self.Y = dict()
        self.attributes["num_labels"] = -1
        self.attributes["label_vocab"] = None
        self.attributes["task_type"] = "seq_tagging"

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        utf8_type = h5py.string_dtype("utf-8", None)
        for key in self.X.keys():
            f["X/" + str(key)] = np.array(self.X[key], dtype=utf8_type)
            f["Y/" + str(key)] = np.array(self.Y[key], dtype=utf8_type)
        f.close()


class Seq2SeqRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super(Seq2SeqRawDataLoader, self).__init__(data_path)
        self.X = dict()
        self.Y = dict()
        self.task_type = "seq2seq"

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.X.keys():
            f["X/" + str(key)] = self.X[key]
            f["Y/" + str(key)] = self.Y[key]
        f.close()


class LanguageModelRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path):
        super(LanguageModelRawDataLoader, self).__init__(data_path)
        self.X = dict()
        self.task_type = "lm"

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in tqdm(self.X.keys(), desc="generate data h5 file"):
            f["X/" + str(key)] = self.X[key]
        f.close()
