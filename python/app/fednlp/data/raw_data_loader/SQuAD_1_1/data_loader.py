import json
import os
import random
import re
import nltk
import h5py


from fedml.data.fednlp.base.raw_data.base_raw_data_loader import SpanExtractionRawDataLoader


class RawDataLoader(SpanExtractionRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_file_name = "train-v1.1.json"
        self.test_file_name = "dev-v1.1.json"
        self.question_ids = dict()

    def load_data(self):
        if len(self.context_X) == 0 or len(self.question_X) == 0 or len(self.Y) == 0:
            self.attributes["doc_index"] = dict()
            train_size = self.process_data_file(
                os.path.join(self.data_path, self.train_file_name)
            )
            test_size = self.process_data_file(
                os.path.join(self.data_path, self.test_file_name)
            )
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [
                i for i in range(train_size, train_size + test_size)
            ]
            self.attributes["index_list"] = (
                self.attributes["train_index_list"] + self.attributes["test_index_list"]
            )

    def process_data_file(self, file_path):
        cnt = 0
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for doc_idx, document in enumerate(data["data"]):
                for paragraph in document["paragraphs"]:
                    for qas in paragraph["qas"]:
                        for answer in qas["answers"]:
                            assert (
                                len(self.context_X)
                                == len(self.question_X)
                                == len(self.Y)
                                == len(self.question_ids)
                            )
                            idx = len(self.context_X)
                            self.context_X[idx] = paragraph["context"]
                            self.question_X[idx] = qas["question"]
                            start = answer["answer_start"]
                            self.Y_answer[idx] = answer["text"]
                            end = start + len(answer["text"].rstrip())
                            self.Y[idx] = (start, end)
                            self.question_ids[idx] = qas["id"]
                            self.attributes["doc_index"][idx] = doc_idx
                            cnt += 1

        return cnt

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.context_X.keys():
            f["context_X/" + str(key)] = self.context_X[key]
            f["question_X/" + str(key)] = self.question_X[key]
            f["Y/" + str(key)] = self.Y[key]
            f["question_ids/" + str(key)] = self.question_ids[key]
            f["Y_answer/" + str(key)] = self.Y_answer[key]
        f.close()
