import os
import sys
import random
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.partition import *


class DataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.task_type = "dialog_response_generation"
        self.conversations = []
        self.attributes = None
        self.movie_conversation_file_name = "movie_conversations.txt"
        self.movie_line_file_name = "movie_lines.txt"

    def data_loader(self):
        if len(self.conversations) == 0:
            conversations, attributes = self.process_data(self.data_path)
            self.conversations, self.attributes = conversations, attributes

        return {"conversations": self.conversations, "attributes": self.attributes,
                "task_type": self.task_type}

    def process_data(self, file_path):
        line_dict = {}
        with open(os.path.join(file_path, self.movie_line_file_name), "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = line.split("+++$+++")
                    line_dict[temp[0].strip()] = temp[-1].strip()

        conversations = []
        single_conversation = []
        attributes = dict()
        attributes["characters"] = []
        attributes["movie"] = []

        with open(os.path.join(file_path, self.movie_conversation_file_name), 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = line.split("+++$+++")
                    conversation_idx = temp[-1].strip()
                    conversation_idx = eval(conversation_idx)
                    for i in range(len(conversation_idx)):
                        attributes["movie"].append(temp[2])
                        attributes["characters"].append((temp[0], temp[1]))
                        single_conversation.append(line_dict[conversation_idx[i]])
                    conversations.append(single_conversation.copy())
                    single_conversation.clear()
        return conversations, attributes

    # TODO: Unified Partition Interface
    @staticmethod
    def nature_partition(attributes):
        movie_set = set(attributes["movie"])
        partition_dict = dict()
        partition_dict["n_clients"] = len(movie_set)
        partition_dict["partition_data"] = dict()
        for i, movie_id in enumerate(movie_set):
            for j in range(len(attributes["movie"])):
                if attributes["movie"][i] == movie_id:
                    if i not in partition_dict["partition_data"]:
                        partition_dict["partition_data"][i] = dict()
                        partition_dict["partition_data"][i]["train"] = list()
                        partition_dict["partition_data"][i]["test"] = list()
                    else:
                        partition_dict["partition_data"][i]["train"].append(j)
        for client_id in partition_dict["partition_data"].keys():
            train_set = partition_dict["partition_data"][client_id]["train"]
            random.shuffle(train_set)
            train_num = int(len(train_set) * 0.8)
            partition_dict["partition_data"][client_id]["train"] = train_set[:train_num]
            partition_dict["partition_data"][client_id]["test"] = train_set[train_num:]
        return partition_dict


if __name__ == "__main__":
    import pickle
    train_file_path = "../../../../data/fednlp/seq2seq/CornellMovieDialogue/cornell movie-dialogs corpus/"
    data_loader = DataLoader(train_file_path)
    train_data_loader = data_loader.data_loader()
    nature_partition_dict = DataLoader.nature_partition(train_data_loader["attributes"])
    uniform_partition_dict = uniform_partition([train_data_loader["conversations"]])

    # pickle.dump(train_data_loader, open("cornell_movie_dialogue_data_loader.pkl", "wb"))
    # pickle.dump({"uniform_partition": uniform_partition_dict, "nature_partition": nature_partition_dict},
    #             open("cornell_movie_dialogue_partition.pkl", "wb"))
    print("done")
