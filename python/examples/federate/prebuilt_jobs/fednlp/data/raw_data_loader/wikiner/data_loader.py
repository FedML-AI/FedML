from fedml.data.fednlp.base.raw_data.base_raw_data_loader import SeqTaggingRawDataLoader
import os
import h5py
import json
import numpy as np


class RawDataLoader(SeqTaggingRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.wp2_data_path = "aij-wikiner-en-wp2"
        self.wp3_data_path = "aij-wikiner-en-wp3"
        self.all_deps = dict()

    def load_data(self):
        if (
            len(self.X) == 0
            or len(self.Y) == 0
            or self.attributes["label_vocab"] is None
        ):
            total_size = self.process_data_file(
                os.path.join(self.data_path, self.wp2_data_path)
            )
            total_size += self.process_data_file(
                os.path.join(self.data_path, self.wp3_data_path)
            )
            self.attributes["index_list"] = [i for i in range(total_size)]
            self.attributes["label_vocab"] = dict()
            for labels in self.Y.values():
                for label in labels:
                    if label not in self.attributes["label_vocab"]:
                        self.attributes["label_vocab"][label] = len(
                            self.attributes["label_vocab"]
                        )

    def process_data_file(self, file_path):
        cnt = 0
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i != 0:
                    line = line.strip()
                    if line:
                        single_x = []
                        single_y = []
                        single_dep = []
                        tokens = line.split(" ")
                        for token in tokens:
                            word, dep, label = token.split("|")
                            single_x.append(word)
                            single_y.append(label)
                            single_dep.append(dep)
                        assert len(self.X) == len(self.Y) == len(self.all_deps)
                        idx = len(self.X)
                        self.X[idx] = single_x
                        self.Y[idx] = single_y
                        self.all_deps[idx] = single_dep
                        cnt += 1
        return cnt

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        utf8_type = h5py.string_dtype("utf-8", None)
        for key in self.X.keys():
            f["X/" + str(key)] = np.array(self.X[key], dtype=utf8_type)
            f["Y/" + str(key)] = np.array(self.Y[key], dtype=utf8_type)
            f["all_deps/" + str(key)] = np.array(self.all_deps[key], dtype=utf8_type)
        f.close()
