# import pickle

#
# from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader
# from data_preprocessing.base.partition import *


# class RawDataLoader(BaseRawDataLoader):
#     def __init__(self, data_path):
#         super().__init__(data_path)
#         self.task_type = "machine_translation"

#     def data_loader(self):
#         if len(self.X) == 0 or len(self.Y) == 0:
#             X, Y = self.process_data(self.data_path)
#             self.X = {i: d for i, d in enumerate(X)}
#             self.Y = {i: d for i, d in enumerate(Y)}
#             index_list = [i for i in range(len(self.X))]
#             self.attributes = {"index_list": index_list}
#         return {"X": self.X, "Y": self.Y, "task_type": self.task_type, "attributes": self.attributes}

#     def process_data(self, file_path):
#         source_file_path = file_path[0]
#         target_file_path = file_path[1]
#         X = []
#         Y = []
#         with open(source_file_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 X.append(line)
#         with open(target_file_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 Y.append(line)
#         return X, Y


# class ClientDataLoader(BaseClientDataLoader):
#     def __init__(self, data_path, partition_path, language_pair, client_idx=None, partition_method="uniform",
#                  tokenize=False):
#         data_fields = ["X", "Y"]
#         super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
#         self.language_pair = language_pair
#         if self.tokenize:
#             self.tokenize_data()

#     def tokenize_data(self):
#         source_tokenizer = self.spacy_tokenizer[self.language_pair[0] + "_tokenizer"]
#         target_tokenizer = self.spacy_tokenizer[self.language_pair[1] + "_tokenizer"]

#         def __tokenize_data(data):
#             for i in range(len(data["X"])):
#                 data["X"][i] = [token.text.strip().lower() for token in source_tokenizer(data["X"][i].strip()) if token.text.strip()]
#                 data["Y"][i] = [token.text.strip().lower() for token in target_tokenizer(data["Y"][i].strip()) if token.text.strip()]

#         __tokenize_data(self.train_data)
#         __tokenize_data(self.test_data)


# # if __name__ == "__main__":
# #     data_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.cs",
# #                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.en"]
# #     data_loader = RawDataLoader(data_file_paths)
# #     results = data_loader.data_loader()
# #     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
# #     pickle.dump(results, open("wmt_cs_en_data_loader.pkl", "wb"))
# #     pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_cs_en_partition.pkl", "wb"))

# #     data_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.de",
# #                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.de-en.en"]
# #     data_loader = RawDataLoader(data_file_paths)
# #     results = data_loader.data_loader()
# #     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
# #     pickle.dump(results, open("wmt_de_en_data_loader.pkl", "wb"))
# #     pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_de_en_partition.pkl", "wb"))

# #     data_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.ru",
# #                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.ru-en.en"]
# #     data_loader = RawDataLoader(data_file_paths)
# #     results = data_loader.data_loader()
# #     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
# #     pickle.dump(results, open("wmt_ru_en_data_loader.pkl", "wb"))
# #     pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_ru_en_partition.pkl", "wb"))

# #     data_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.zh",
# #                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.zh-en.en"]
# #     data_loader = RawDataLoader(data_file_paths)
# #     results = data_loader.data_loader()
# #     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
# #     pickle.dump(results, open("wmt_zh_en_data_loader.pkl", "wb"))
# #     pickle.dump({"uniform_partition": uniform_partition_dict}, open("wmt_zh_en_partition.pkl", "wb"))
# #     print("done")
