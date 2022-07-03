import os
import random
import h5py
import numpy as np
import json


from data.raw_data_loader.base.base_raw_data_loader import Seq2SeqRawDataLoader


class RawDataLoader(Seq2SeqRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.history = dict()
        self.movie_conversation_file_name = "movie_conversations.txt"
        self.movie_line_file_name = "movie_lines.txt"

    def load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0 or len(self.history) == 0:
            total_size = self.process_data_file(self.data_path)
            self.attributes["index_list"] = [i for i in range(total_size)]

    def process_data_file(self, file_path):
        line_dict = {}
        with open(
            os.path.join(file_path, self.movie_line_file_name), "r", errors="ignore"
        ) as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = line.split("+++$+++")
                    line_dict[temp[0].strip()] = {
                        "utterance": temp[-1].strip(),
                        "character": temp[1],
                    }

        self.attributes["characters"] = dict()
        self.attributes["movie"] = dict()

        conversation = []
        cnt = 0

        with open(os.path.join(file_path, self.movie_conversation_file_name), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = line.split("+++$+++")
                    conversation_idx = temp[-1].strip()
                    conversation_idx = eval(conversation_idx)
                    for i in range(len(conversation_idx) - 1):
                        assert len(self.X) == len(self.Y) == len(self.history)
                        idx = len(self.X)
                        self.X[idx] = line_dict[conversation_idx[i]]["utterance"]
                        self.Y[idx] = line_dict[conversation_idx[i + 1]]["utterance"]
                        self.history[idx] = conversation.copy()
                        self.attributes["movie"][idx] = temp[2]
                        self.attributes["characters"][idx] = (
                            line_dict[conversation_idx[i]]["character"],
                            line_dict[conversation_idx[i + 1]]["character"],
                        )
                        conversation.append(line_dict[conversation_idx[i]]["utterance"])
                        cnt += 1
                    conversation.clear()
        return cnt

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        utf8_type = h5py.string_dtype("utf-8", None)
        for key in self.X.keys():
            f["X/" + str(key)] = self.X[key]
            f["Y/" + str(key)] = self.Y[key]
            f["history/" + str(key)] = np.array(self.history[key], dtype=utf8_type)
        f.close()
