# import os
# import random

#
# from data_preprocessing.base.base_raw_data_loader import BaseRawDataLoader


# class RawDataLoader(BaseRawDataLoader):
#     def __init__(self, data_path):
#         super().__init__(data_path)
#         self.task_type = "dialog_response_generation"
#         self.history = []
#         self.attributes = None
#         self.movie_conversation_file_name = "movie_conversations.txt"
#         self.movie_line_file_name = "movie_lines.txt"

#     def data_loader(self):
#         if len(self.history) == 0:
#             X, Y, history, attributes = self.process_data(self.data_path)
#             self.X = {i: d for i, d in enumerate(X)}
#             self.Y = {i: d for i, d in enumerate(Y)}
#             self.history = {i: d for i, d in enumerate(history)}
#             self.attributes = attributes
#             self.index_list = [i for i in range(len(self.X))]
#             self.attributes["index_list"] = self.index_list
#         return {"X": self.X, "Y": self.Y, "history": self.history, "attributes": self.attributes,
#                 "task_type": self.task_type}

#     def process_data(self, file_path):
#         line_dict = {}
#         with open(os.path.join(file_path, self.movie_line_file_name), "r", errors="ignore") as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     temp = line.split("+++$+++")
#                     line_dict[temp[0].strip()] = {"utterance": temp[-1].strip(), "character": temp[1]}

#         attributes = dict()
#         attributes["characters"] = []
#         attributes["movie"] = []

#         conversation = []
#         X = []
#         Y = []
#         history = []

#         with open(os.path.join(file_path, self.movie_conversation_file_name), 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     temp = line.split("+++$+++")
#                     conversation_idx = temp[-1].strip()
#                     conversation_idx = eval(conversation_idx)
#                     for i in range(len(conversation_idx) - 1):
#                         X.append(line_dict[conversation_idx[i]]["utterance"])
#                         Y.append(line_dict[conversation_idx[i + 1]]["utterance"])
#                         history.append(conversation.copy())
#                         attributes["movie"].append(temp[2])
#                         attributes["characters"].append((line_dict[conversation_idx[i]]["character"],
#                                                          line_dict[conversation_idx[i + 1]]["character"]))
#                         conversation.append(line_dict[conversation_idx[i]]["utterance"])
#                     conversation.clear()
#         return X, Y, history, attributes

#     # TODO: Unified Partition Interface
#     @staticmethod
#     def nature_partition(attributes):
#         movie_set = set(attributes["movie"])
#         partition_dict = dict()
#         partition_dict["n_clients"] = len(movie_set)
#         partition_dict["partition_data"] = dict()
#         for i, movie_id in enumerate(movie_set):
#             for j in range(len(attributes["movie"])):
#                 if attributes["movie"][j] == movie_id:
#                     if i not in partition_dict["partition_data"]:
#                         partition_dict["partition_data"][i] = dict()
#                         partition_dict["partition_data"][i]["train"] = list()
#                         partition_dict["partition_data"][i]["test"] = list()
#                     else:
#                         partition_dict["partition_data"][i]["train"].append(j)
#         for client_id in partition_dict["partition_data"].keys():
#             train_set = partition_dict["partition_data"][client_id]["train"]
#             random.shuffle(train_set)
#             train_num = int(len(train_set) * 0.8)
#             partition_dict["partition_data"][client_id]["train"] = train_set[:train_num]
#             partition_dict["partition_data"][client_id]["test"] = train_set[train_num:]
#         return partition_dict


# class ClientDataLoader(BaseClientDataLoader):

#     def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
#         data_fields = ["X", "Y", "history"]
#         super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
#         if self.tokenize:
#             self.tokenize_data()

#     def tokenize_data(self):
#         tokenizer = self.spacy_tokenizer.en_tokenizer

#         def __tokenize_data(data):
#             for i in range(len(data["X"])):
#                 data["X"][i] = [token.text.strip().lower() for token in tokenizer(data["X"][i].strip()) if token.text.strip()]
#                 data["Y"][i] = [token.text.strip().lower() for token in tokenizer(data["Y"][i].strip()) if token.text.strip()]
#                 for j in range(len(data["history"][i])):
#                     data["history"][i][j] = [token.text.strip().lower() for token in tokenizer(data["history"][i][j].strip()) if
#                                     token.text.strip()]

#         __tokenize_data(self.train_data)
#         __tokenize_data(self.test_data)

# # if __name__ == "__main__":
# #     data_file_path = "../../../../data/fednlp/seq2seq/CornellMovieDialogue/cornell_movie_dialogs_corpus/"
# #     data_loader = RawDataLoader(data_file_path)
# #     results = data_loader.data_loader()
# #     nature_partition_dict = RawDataLoader.nature_partition(results["attributes"])
# #     uniform_partition_dict = uniform_partition(results["attributes"]["index_list"])
# #
# #     pickle.dump(train_data_loader, open("cornell_movie_dialogue_data_loader.pkl", "wb"))
# #     pickle.dump({"uniform": uniform_partition_dict, "nature": nature_partition_dict},
# #                 open("cornell_movie_dialogue_partition.pkl", "wb"))
# #     print("done")
