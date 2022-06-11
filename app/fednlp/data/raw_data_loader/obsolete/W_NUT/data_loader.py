# import os

#
# from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader
# from data_preprocessing.base.utils import build_vocab


# class RawDataLoader(BaseRawDataLoader):
#     def __init__(self, data_path):
#         super().__init__(data_path)
#         self.task_type = "sequence_tagging"
#         self.target_vocab = None
#         self.train_file_name = "wnut17train.conll"
#         self.test_file_name = "emerging.test.annotated"

#     def data_loader(self):
#         if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
#             X, Y = self.process_data(os.path.join(self.data_path, self.train_file_name))
#             train_size = len(X)
#             temp = self.process_data(os.path.join(self.data_path, self.test_file_name))
#             X.extend(temp[0])
#             Y.extend(temp[1])
#             self.X = {i: d for i, d in enumerate(X)}
#             self.Y = {i: d for i, d in enumerate(Y)}
#             train_index_list = [i for i in range(train_size)]
#             test_index_list = [i for i in range(train_size, len(self.X))]
#             index_list = train_index_list + test_index_list
#             self.target_vocab = build_vocab(Y)
#             self.attributes = {"train_index_list": train_index_list, "test_index_list": test_index_list,
#                                "index_list": index_list, "target_vocab": self.target_vocab}
#         return {"X": self.X, "Y": self.Y, "task_type": self.task_type,
#                 "attributes": self.attributes}

#     def process_data(self, file_path):
#         X = []
#         Y = []
#         single_x = []
#         single_y = []
#         with open(file_path, 'r', encoding="utf8") as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     token, label = line.split("\t")
#                     single_x.append(token)
#                     single_y.append(label)
#                 else:
#                     if len(single_x) != 0:
#                         X.append(single_x.copy())
#                         Y.append(single_y.copy())
#                     single_x.clear()
#                     single_y.clear()
#         return X, Y


# class ClientDataLoader(BaseClientDataLoader):
#     def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
#         data_fields = ["X", "Y"]
#         super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
