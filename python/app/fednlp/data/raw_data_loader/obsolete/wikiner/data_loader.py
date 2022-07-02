#
# from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader
# from data_preprocessing.base.utils import *
# import os


# class RawDataLoader(BaseRawDataLoader):
#     def __init__(self, data_path):
#         super().__init__(data_path)
#         self.task_type = "sequence_tagging"
#         self.wp2_data_path = "aij-wikiner-en-wp2"
#         self.wp3_data_path = "aij-wikiner-en-wp3"
#         self.target_vocab = None
#         self.all_deps = None

#     def data_loader(self):
#         if len(self.X) == 0 or len(self.Y) == 0 or len(self.target_vocab) == 0:
#             X, Y, all_deps = self.process_data(os.path.join(self.data_path, self.wp2_data_path))
#             temp = self.process_data(os.path.join(self.data_path, self.wp3_data_path))
#             X.extend(temp[0])
#             Y.extend(temp[1])
#             all_deps.extend(temp[2])
#             self.X = {i: d for i, d in enumerate(X)}
#             self.Y = {i: d for i, d in enumerate(Y)}
#             self.all_deps = {i: d for i, d in enumerate(all_deps)}
#             index_list = [i for i in range(len(self.X))]
#             self.target_vocab = build_vocab(Y)
#             self.attributes = {"index_list": index_list, "target_vocab": self.target_vocab}
#         return {"X": self.X, "Y": self.Y, "all_deps": self.all_deps,
#                 "task_type": self.task_type, "attributes": self.attributes}

#     def process_data(self, file_path):
#         X = []
#         Y = []
#         all_deps = []
#         with open(file_path, "r") as f:
#             for i, line in enumerate(f):
#                 if i != 0:
#                     line = line.strip()
#                     if line:
#                         single_x = []
#                         single_y = []
#                         single_dep = []
#                         tokens = line.split(" ")
#                         for token in tokens:
#                             word, dep, label = token.split("|")
#                             single_x.append(word)
#                             single_y.append(label)
#                             single_dep.append(dep)
#                         X.append(single_x)
#                         Y.append(single_y)
#                         all_deps.append(single_dep)
#         return X, Y, all_deps


# class ClientDataLoader(BaseClientDataLoader):
#     def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
#         data_fields = ["X", "Y"]
#         super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
