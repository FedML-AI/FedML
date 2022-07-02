# import json
# import os
# import random
# import re
# import nltk

#
# from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader


# class RawDataLoader(BaseRawDataLoader):
#     def __init__(self, data_path):
#         super().__init__(data_path)
#         self.task_type = "span_extraction"
#         self.document_X = []
#         self.question_X = []
#         self.attributes = dict()
#         self.train_file_name = "train-v1.1.json"
#         self.test_file_name = "dev-v1.1.json"

#     def data_loader(self):
#         if len(self.document_X) == 0 or len(self.question_X) == 0 or len(self.Y) == 0:
#             context_X, question_X, Y, question_ids = self.process_data(os.path.join(self.data_path, self.train_file_name))
#             train_size = len(context_X)
#             temp = self.process_data(os.path.join(self.data_path, self.test_file_name))
#             context_X.extend(temp[0])
#             question_X.extend(temp[1])
#             Y.extend(temp[2])
#             question_ids.extend(temp[3])
#             train_index_list = [i for i in range(train_size)]
#             test_index_list = [i for i in range(train_size, len(context_X))]
#             index_list = train_index_list + test_index_list
#             self.context_X = {i: d for i, d in enumerate(context_X)}
#             self.question_X = {i: d for i, d in enumerate(question_X)}
#             self.question_ids = {i: d for i, d in enumerate(question_ids)}
#             self.Y = {i: d for i, d in enumerate(Y)}
#             self.attributes["train_index_list"] = train_index_list
#             self.attributes["test_index_list"] = test_index_list
#             self.attributes["index_list"] = index_list
#         return {"context_X": self.context_X, "question_X": self.question_X, "Y": self.Y, "question_ids": self.question_ids,
#             "attributes": self.attributes, "task_type": self.task_type}

#     def process_data(self, file_path):
#         context_X = []
#         question_X = []
#         Y = []
#         question_ids = []
#         if "doc_index" not in self.attributes:
#             self.attributes["doc_index"] = []
#         with open(file_path, "r", encoding='utf-8') as f:
#             data = json.load(f)

#             for doc_idx, document in enumerate(data["data"]):
#                 for paragraph in document["paragraphs"]:
#                     for qas in paragraph["qas"]:
#                         for answer in qas["answers"]:
#                             context_X.append(paragraph["context"])
#                             question_X.append(qas["question"])
#                             start = answer["answer_start"]
#                             end = start + len(answer["text"].rstrip())
#                             Y.append((start, end))
#                             question_ids.append(qas["id"])
#                             self.attributes["doc_index"].append(doc_idx)

#         return context_X, question_X, Y, question_ids


#     # TODO: Unified Partition Interface
#     @staticmethod
#     def nature_partition(attributes):
#         train_doc_index_set = set([attributes["doc_index"][i] for i in attributes["train_index_list"]])
#         partition_dict = dict()
#         partition_dict["partition_data"] = dict()
#         partition_dict["n_clients"] = len(train_doc_index_set)
#         for doc_id in train_doc_index_set:
#             for i in attributes["train_index_list"]:
#                 if attributes["doc_index"][i] == doc_id:
#                     if doc_id not in partition_dict["partition_data"]:
#                         partition_dict["partition_data"][doc_id] = dict()
#                         partition_dict["partition_data"][doc_id]["train"] = list()
#                         partition_dict["partition_data"][doc_id]["test"] = list()
#                     partition_dict["partition_data"][doc_id]["train"].append(i)

#         test_doc_index_set = set([attributes["doc_index"][i] for i in attributes["test_index_list"]])
#         for doc_id in test_doc_index_set:
#             test_doc_index_list = []
#             for i in attributes["test_index_list"]:
#                 if attributes["doc_index"][i] == doc_id:
#                     test_doc_index_list.append(i)
#             client_idx = random.randint(0, partition_dict["n_clients"] - 1)
#             partition_dict["partition_data"][client_idx]["test"].extend(test_doc_index_list)

#         return partition_dict

# class ClientDataLoader(BaseClientDataLoader):


#     def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False, data_filter=None):
#         data_fields = ["context_X", "question_X", "Y", "question_ids"]
#         super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
#         self.clean_data()
#         if self.tokenize:
#             self.tokenize_data()
#             self.transform_labels()

#         if data_filter:
#             data_filter(self.train_data)
#             data_filter(self.test_data)

#     def clean_data(self):
#         def __clean_data(data):
#             for i in range(len(data["context_X"])):
#                 data["context_X"][i] = data["context_X"][i].replace("''", '" ').replace("``", '" ')
#         __clean_data(self.train_data)
#         __clean_data(self.test_data)

#     def tokenize_data(self):

#         def word_tokenize(sent):
#              return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(sent)]
#         def __tokenize_data(data):
#             data["tokenized_context_X"] = list()
#             data["tokenized_question_X"] = list()
#             data["char_context_X"] = list()
#             data["char_question_X"] = list()
#             self.data_fields.extend(["tokenized_context_X", "tokenized_question_X", "char_context_X", "char_question_X"])
#             for i in range(len(data["context_X"])):
#                 temp_tokens = word_tokenize(data["context_X"][i])
#                 data["tokenized_context_X"].append(self.remove_stop_tokens(temp_tokens))
#                 data["tokenized_question_X"].append(word_tokenize(data["question_X"][i]))
#                 context_chars = [list(token) for token in data["tokenized_context_X"][i]]
#                 question_chars = [list(token) for token in data["tokenized_question_X"][i]]
#                 data["char_context_X"].append(context_chars)
#                 data["char_question_X"].append(question_chars)

#         __tokenize_data(self.train_data)
#         __tokenize_data(self.test_data)

#     def remove_stop_tokens(self, temp_tokens):
#         tokens = []
#         for token in temp_tokens:
#             flag = False
#             l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
#             tokens.extend(re.split("([{}])".format("".join(l)), token))
#         return tokens

#     def transform_labels(self):
#         def __transform_labels(data):
#             for i in range(len(data["context_X"])):
#                 context = data["context_X"][i]
#                 context_tokens = data["tokenized_context_X"][i]
#                 start, stop = data["Y"][i]

#                 spans = self.get_spans(context, context_tokens)
#                 idxs = []
#                 for word_idx, span in enumerate(spans):
#                     if not (stop <= span[0] or start >= span[1]):
#                         idxs.append(word_idx)

#                 data["Y"][i] = (idxs[0], idxs[-1] + 1)
#         __transform_labels(self.train_data)
#         __transform_labels(self.test_data)

#     def get_spans(self, text, all_tokens):
#         spans = []
#         cur_idx = 0
#         for token in all_tokens:
#             if text.find(token, cur_idx) < 0:
#                 print("{} {} {}".format(token, cur_idx, text))
#                 raise Exception()
#             cur_idx = text.find(token, cur_idx)
#             spans.append((cur_idx, cur_idx + len(token)))
#             cur_idx += len(token)
#         return spans


# def get_normal_format(dataset, cut_off=None):
#     """
#     reformat the dataset to normal version.
#     """
#     reformatted_data = []
#     assert len(dataset["context_X"]) == len(dataset["question_X"]) == len(dataset["Y"]) == len(dataset["question_ids"])
#     for c, q, a, qid in zip(dataset["context_X"], dataset["question_X"], dataset["Y"], dataset["question_ids"]):
#         item = {}
#         item["context"] = c
#         item["qas"] = [
#             {
#                 # "id": "%d"%(len(reformatted_data)+1),
#                 "id": qid,
#                 "is_impossible": False,
#                 "question": q,
#                 "answers": [{"text": c[a[0]:a[1]], "answer_start": a[0]}],
#             }
#         ]
#         reformatted_data.append(item)
#     return reformatted_data[:cut_off]
