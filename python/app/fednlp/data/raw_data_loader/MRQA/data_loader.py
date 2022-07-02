from logging import error
import os
import json
import h5py
import string
from numpy.core.arrayprint import repr_format

from data.raw_data_loader.base.base_raw_data_loader import SpanExtractionRawDataLoader

# test script  python test_rawdataloader.py --dataset MRQA --data_dir "../../../../reading_comprehension/" --h5_file_path ../../data_files/mrqa_data.h5


class RawDataLoader(SpanExtractionRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        # i rename some of the file so that they are more distinguishable
        self.train_file_name = [
            "HotpotQA.jsonl",
            "NewsQA.jsonl",
            "SearchQA.jsonl",
            "NaturalQuestionsShort.jsonl",
            "SQuAD.jsonl",
            "TriviaQA.jsonl",
        ]
        self.test_file_name = [
            "HotpotQA-dev.jsonl",
            "NewsQA-dev.jsonl",
            "SearchQA-dev.jsonl",
            "NaturalQuestionsShort-dev.jsonl",
            "SQuAD-dev.jsonl",
            "TriviaQA-dev.jsonl",
        ]
        self.question_ids = dict()
        self.answers = dict()
        self.attributes["train_index_list"] = []
        self.attributes["test_index_list"] = []
        self.attributes["label_index_list"] = []

    def load_data(self):
        if len(self.context_X) == 0 or len(self.question_X) == 0 or len(self.Y) == 0:
            train_size = 0
            test_size = 0
            for train_dataset in self.train_file_name:
                label = train_dataset.split(".")[0]
                train_size += self.process_data_file(
                    os.path.join(self.data_path, train_dataset), label, True
                )
            for test_dataset in self.test_file_name:
                label = test_dataset.split("-")[0]
                test_size += self.process_data_file(
                    os.path.join(self.data_path, test_dataset), label, False
                )
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [
                i for i in range(train_size, train_size + test_size)
            ]
            self.attributes["index_list"] = (
                self.attributes["train_index_list"] + self.attributes["test_index_list"]
            )
            assert len(self.attributes["index_list"]) == len(
                self.attributes["label_index_list"]
            )
            print(len(self.attributes["train_index_list"]))
            print(len(self.attributes["test_index_list"]))

    def process_data_file(self, file_path, label, is_train):
        cnt = 0
        question_cnt = 0
        printable = set(string.printable)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f)
            for line in f:
                paragraph = json.loads(line)
                for question in paragraph["qas"]:
                    if is_train:
                        if question_cnt == 10000:
                            print(cnt)
                            print("finish loading ", file_path)
                            return cnt
                    else:
                        if question_cnt == 500:
                            print(cnt)
                            print("finish loading ", file_path)
                            return cnt
                    seen_answers = set()
                    for answer in question[
                        "detected_answers"
                    ]:  # same answer continue or not?
                        if answer["text"] in seen_answers:
                            continue
                        seen_answers.add(answer["text"])
                        assert (
                            len(self.context_X)
                            == len(self.question_X)
                            == len(self.Y)
                            == len(self.question_ids)
                        )
                        idx = len(self.context_X)
                        # clean context data
                        context = "".join(
                            filter(lambda x: x in printable, paragraph["context"])
                        )
                        actual_start = context.find(answer["text"])
                        if actual_start == -1:
                            continue
                            # print("not exist in "+ file_path+ "for this answer "+ answer["text"] )
                        self.context_X[idx] = context
                        self.question_X[idx] = question["question"]
                        start = actual_start
                        end = actual_start + len(answer["text"])
                        self.Y[idx] = (start, end)
                        self.answers[idx] = answer["text"]
                        self.question_ids[idx] = question["qid"]
                        self.attributes["label_index_list"].append(label)
                        cnt += 1
                    question_cnt += 1
        return cnt

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for key in self.context_X.keys():
            f["context_X/" + str(key)] = self.context_X[key]
            f["question_X/" + str(key)] = self.question_X[key]
            f["Y/" + str(key)] = self.Y[key]
            f["Y_answer/" + str(key)] = self.answers[key]
            f["question_ids/" + str(key)] = self.question_ids[key]
        f.close()
