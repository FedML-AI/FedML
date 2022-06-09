# import os

# from nltk.tree import Tree

#
# from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader


# class RawDataLoader(BaseRawDataLoader):
#     def __init__(self, data_path):
#         super().__init__(data_path)
#         self.task_type = "text_classification"
#         self.target_vocab = None
#         self.train_file_name = "train.txt"
#         self.test_file_name = "test.txt"

#     def data_loader(self):
#         if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
#             X, Y = self.process_data(os.path.join(self.data_path, self.train_file_name))
#             train_size = len(X)
#             temp_X, temp_Y = self.process_data(os.path.join(self.data_path, self.test_file_name))
#             X.extend(temp_X)
#             Y.extend(temp_Y)
#             self.X = {i: d for i, d in enumerate(X)}
#             self.Y = {i: d for i, d in enumerate(Y)}
#             train_index_list = [i for i in range(train_size)]
#             test_index_list = [i for i in range(train_size, len(X))]
#             index_list = train_index_list + test_index_list
#             self.target_vocab = {key: i for i, key in enumerate(set(Y))}
#             self.attributes = {"index_list": index_list, "train_index_list": train_index_list,
#                                "test_index_list": test_index_list, "target_vocab": self.target_vocab}
#         return {"X": self.X, "Y": self.Y, "task_type": self.task_type,
#                 "attributes": self.attributes}

#     def label_level(self, label):
#         return {'0': 'negative', '1': 'negative', '2': 'neutral',
#                     '3': 'positive', '4': 'positive', None: None}[label]

#     def process_data(self, file_path):
#         X = []
#         Y = []
#         with open(file_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 tree = Tree.fromstring(line)
#                 label = self.label_level(tree.label())
#                 if label != "neutral":
#                     X.append(" ".join(tree.leaves()))
#                     Y.append(label)
#         return X, Y

# class ClientDataLoader(BaseClientDataLoader):

#     def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
#         data_fields = ["X", "Y"]
#         super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
#         if self.tokenize:
#             self.tokenize_data()

#     def tokenize_data(self):

#         def __tokenize_data(data):
#             for i in range(len(data["X"])):
#                 data["X"][i] = data["X"][i].split(" ")

#         __tokenize_data(self.train_data)
#         __tokenize_data(self.test_data)
