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

class DataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "span_extraction"
        self.document_X = []
        self.question_X = []

    def data_loader(self):
        if len(self.document_X) == 0 or len(self.question_X) == 0 or len(self.Y) == 0:
            context_X, question_X, Y, attributes = self.process_data(self.data_path)
            self.context_X, self.question_X, self.Y, self.attributes = context_X, question_X, Y, attributes
        return {"context_X": self.context_X, "question_X": self.question_X, "Y": self.Y, "attributes": self.attributes,
                "task_type": self.task_type}

    def process_data(self, file_path):
        context_X = []
        question_X = []
        Y = []
        attributes = dict()
        attributes["doc_index"] = []
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)

            for index, document in enumerate(data["data"]):
                for paragraph in document["paragraphs"]:
                    for qas in paragraph["qas"]:
                        answers_index = []
                        answers_text = []
                        context_X.append(paragraph["context"])
                        question_X.append(qas["question"])
                        for answer in qas["answers"]:
                            if answer["text"] not in answers_text:
                                answers_text.append(answer["text"])
                                start = answer["answer_start"]
                                end = start + len(answer["text"].rstrip())
                                answers_index.append((start, end))
                        Y.append(answers_index)
                        attributes["doc_index"].append(index)

        return context_X, question_X, Y, attributes

    # TODO: Unified Partition Interface
    @staticmethod
    def nature_partition(train_attributes, test_attributes):
        train_doc_index_set = set(train_attributes["doc_index"])
        partition_dict = dict()
        partition_dict["partition_data"] = dict()
        partition_dict["n_clients"] = len(train_doc_index_set)
        for doc_id in train_doc_index_set:
            for i in range(len(train_attributes["doc_index"])):
                if train_attributes["doc_index"][i] == doc_id:
                    if doc_id not in partition_dict["partition_data"]:
                        partition_dict["partition_data"][doc_id] = dict()
                        partition_dict["partition_data"][doc_id]["train"] = list()
                        partition_dict["partition_data"][doc_id]["test"] = list()
                    else:
                        partition_dict["partition_data"][doc_id]["train"].append(i)

        test_doc_index_set = set(test_attributes["doc_index"])
        for doc_id in test_doc_index_set:
            test_doc_index_list = []
            for i in range(len(test_attributes["doc_index"])):
                if test_attributes["doc_index"][i] == doc_id:
                    test_doc_index_list.append(i + len(train_attributes["doc_index"]))
            client_idx = random.randint(0, partition_dict["n_clients"] - 1)
            partition_dict["partition_data"][client_idx]["test"].extend(test_doc_index_list)

        return partition_dict





if __name__ == "__main__":
    import pickle
    data_path = '../../../../data/fednlp/span_extraction/SQuAD_1.1'
    train_file_path = '../../../../data/fednlp/span_extraction/SQuAD_1.1/train-v1.1.json'
    test_file_path = '../../../../data/fednlp/span_extraction/SQuAD_1.1/dev-v1.1.json'

    train_data_loader = DataLoader(train_file_path)
    train_result = train_data_loader.data_loader()

    test_data_loader = DataLoader(test_file_path)
    test_result = test_data_loader.data_loader()

    nature_partition = DataLoader.nature_partition(train_result["attributes"], test_result["attributes"])
    uniform_partition_dict = uniform_partition([train_result["context_X"], train_result["question_X"], train_result["Y"]],
                                               [test_result["context_X"], test_result["question_X"], test_result["Y"]])

    # pickle_dict = train_result
    # pickle_dict["context_X"].extend(test_result["context_X"])
    # pickle_dict["question_X"].extend(test_result["question_X"])
    # pickle_dict["Y"].extend(test_result["Y"])
    # pickle_dict["attributes"]["doc_index"].extend(
    #     [idx+len(set(train_result["attributes"]["doc_index"])) for idx in test_result["attributes"]["doc_index"]])
    # pickle_dict["attributes"]["train"] = [i for i in range(len(train_result["context_X"]))]
    # pickle_dict["attributes"]["test"] = [i + len(train_result["context_X"]) for i in range(len(test_result["context_X"]))]
    #
    # pickle.dump(pickle_dict, open("squad_1.1_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict, "nature_partition": nature_partition},
    #             open("squad_1.1_partition.pkl", "wb"))

    print("done")
