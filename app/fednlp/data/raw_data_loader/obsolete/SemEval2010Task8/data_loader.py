# import os
# import re

#
# from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader


# class RawDataLoader(BaseRawDataLoader):
#     def __init__(self, data_path):
#         super().__init__(data_path)
#         self.task_type = "text_classification"
#         self.target_vocab = None
#         self.train_file_name = "SemEval2010_task8_training/TRAIN_FILE.TXT"
#         self.test_file_name = "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

#     def data_loader(self):
#         if len(self.X) == 0 or len(self.Y) == 0 or self.target_vocab is None:
#             X, Y = self.process_data(os.path.join(self.data_path, self.train_file_name))
#             train_size = len(X)
#             temp = self.process_data(os.path.join(self.data_path, self.test_file_name))
#             X.extend(temp[0])
#             Y.extend(temp[1])
#             self.X = {i: d for i, d in enumerate(X)}
#             self.Y = {i: d for i, d in enumerate(Y)}
#             self.target_vocab = {key: i for i, key in enumerate(set(Y))}
#             train_index_list = [i for i in range(train_size)]
#             test_index_list = [i for i in range(train_size, len(self.X))]
#             index_list = train_index_list + test_index_list
#             self.attributes = {"train_index_list": train_index_list, "test_index_list": test_index_list,
#                                "index_list": index_list, "target_vocab": self.target_vocab}
#         return {"X": self.X, "Y": self.Y, "task_type": self.task_type,
#                 "attributes": self.attributes}

#     def process_data(self, file_path):
#         X = []
#         Y = []
#         print(file_path)
#         with open(file_path, "r", encoding='utf-8') as f:
#             data = f.readlines()
#             for i in range(len(data)):
#                 if len(data[i]) > 1 and data[i][0].isdigit():
#                     clean_data = data[i].split('\t')[1][1:-1].strip()
#                     X.append(clean_data)

#                 elif len(data[i - 1]) > 1 and data[i - 1][0].isdigit():
#                     label = data[i].rstrip("\n")
#                     Y.append(label)
#         return X, Y


# class ClientDataLoader(BaseClientDataLoader):

#     def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
#         data_fields = ["X", "Y"]
#         super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
#         if self.tokenize:
#             self.clean_and_tokenize_data()

#     def clean_and_tokenize_data(self):
#         tokenizer = self.spacy_tokenizer.en_tokenizer

#         def __clean_and_tokenize_data(data):
#             for i in range(len(data["X"])):
#                 sentence = data["X"][i]
#                 e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
#                 e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]
#                 sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
#                 sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
#                 sentence = [token.text.strip() for token in tokenizer(sentence) if token.text.strip()]
#                 sentence = ' '.join(sentence)
#                 sentence = sentence.replace('< e1 >', '<e1>')
#                 sentence = sentence.replace('< e2 >', '<e2>')
#                 sentence = sentence.replace('< /e1 >', '</e1>')
#                 sentence = sentence.replace('< /e2 >', '</e2>')
#                 sentence = sentence.split()
#                 data["X"][i] = sentence

#         __clean_and_tokenize_data(self.train_data)
#         __clean_and_tokenize_data(self.test_data)
