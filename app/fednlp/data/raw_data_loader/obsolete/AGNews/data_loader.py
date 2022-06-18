# import csv
# import os

#
# from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader


# class RawDataLoader(BaseRawDataLoader):
#     def __init__(self, data_path):
#         super().__init__(data_path)
#         self.task_type = "text_classification"
#         self.target_vocab = None
#         self.train_path = "train.csv"
#         self.test_path = "test.csv"

#     def data_loader(self):
#         if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
#             X, Y = self.process_data(os.path.join(self.data_path, self.train_path))
#             train_size = len(X)
#             temp = self.process_data(os.path.join(self.data_path, self.test_path))
#             X.extend(temp[0])
#             Y.extend(temp[1])
#             self.X = {i: d for i, d in enumerate(X)}
#             self.Y = {i: d for i, d in enumerate(Y)}
#             self.target_vocab = {key: i for i, key in enumerate(set(Y))}
#             train_index_list = [i for i in range(train_size)]
#             test_index_list = [i for i in range(train_size, len(X))]
#             index_list = train_index_list + test_index_list
#             self.attributes = {"index_list": index_list, "train_index_list": train_index_list,
#                                "test_index_list": test_index_list, "target_vocab": self.target_vocab}
#         return {"X": self.X, "Y": self.Y, "task_type": self.task_type,
#                 "attributes": self.attributes}

#     def process_data(self, file_path):
#         X = []
#         Y = []
#         with open(file_path, "r", newline='') as csvfile:
#             data = csv.reader(csvfile, delimiter=',')
#             for line in data:
#                 target = line[0]
#                 source = line[2].replace('\\', '')
#                 X.append(source)
#                 Y.append(target)
#         return X, Y


# class ClientDataLoader(BaseClientDataLoader):
#     def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
#         data_fields = ["X", "Y"]
#         super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
#         if self.tokenize:
#             self.tokenize_data()

#     def tokenize_data(self):
#         tokenizer = self.spacy_tokenizer.en_tokenizer

#         def __tokenize_data(data):
#             for i in range(len(data["X"])):
#                 data["X"][i] = [token.text.strip().lower() for token in tokenizer(data["X"][i].strip()) if token.text.strip()]

#         __tokenize_data(self.train_data)
#         __tokenize_data(self.test_data)
