# import os

#
# from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader


# class RawDataLoader(BaseRawDataLoader):
#     def __init__(self, data_path):
#         super().__init__(data_path)
#         self.task_type = "summarization"
#         self.cnn_path = "cnn/stories"
#         self.dailymail_path = "dailymail/stories"

#     def data_loader(self):
#         if len(self.X) == 0 or len(self.Y) == 0:
#             X = None
#             Y = None
#             for root, dirs, files in os.walk(os.path.join(self.data_path, self.cnn_path)):
#                 for file_name in files:
#                     file_path = os.path.join(root, file_name)
#                     if X is None or Y is None:
#                         X, Y = self.process_data(file_path)
#                     else:
#                         temp = self.process_data(file_path)
#                         X.extend(temp[0])
#                         Y.extend(temp[1])
#             for root, dirs, files in os.walk(os.path.join(self.data_path, self.dailymail_path)):
#                 for file_name in files:
#                     file_path = os.path.join(root, file_name)
#                     temp = self.process_data(file_path)
#                     X.extend(temp[0])
#                     Y.extend(temp[1])
#             self.X = {i: d for i, d in enumerate(X)}
#             self.Y = {i: d for i, d in enumerate(Y)}
#             index_list = [i for i in range(len(self.X))]
#             self.attributes = {"index_list": index_list}
#         return {"X": self.X, "Y": self.Y, "task_type": self.task_type, "attributes": self.attributes}

#     def process_data(self, file_path):
#         X = []
#         Y = []
#         article_lines = []
#         abstract_lines = []
#         next_is_highlight = False
#         with open(file_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     if line.startswith("@highlight"):
#                         next_is_highlight = True
#                     elif next_is_highlight:
#                         abstract_lines.append(line)
#                     else:
#                         article_lines.append(line)
#         X.append(" ".join(article_lines))
#         Y.append(' '.join(["%s %s %s" % ("<s>", sent, "</s>") for sent in abstract_lines]))
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
#                 data["Y"][i] = [token.text.strip().lower() for token in tokenizer(data["Y"][i].strip()) if token.text.strip()]

#         __tokenize_data(self.train_data)
#         __tokenize_data(self.test_data)
