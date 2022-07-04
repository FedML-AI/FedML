import os
import h5py
import json

# temp
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from data.raw_data_loader.base.base_raw_data_loader import LanguageModelRawDataLoader
from tqdm import tqdm
import math
from multiprocessing import Pool
import logging

global_X = None

CHUNK_SIZE = 1000000

logging.basicConfig(
    level=logging.INFO,
    format="%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)


class RawDataLoader(LanguageModelRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_file_name = "stackoverflow_train.h5"
        self.test_file_name = "stackoverflow_test.h5"
        self._EXAMPLE = "examples"
        self._TOKENS = "tokens"
        self.nature_partition_dict = dict()

    def load_data(self):
        if len(self.X) == 0:
            train_size = self.process_data_file(
                os.path.join(self.data_path, self.train_file_name)
            )
            test_size = self.process_data_file(
                os.path.join(self.data_path, self.test_file_name), test=True
            )
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [
                i for i in range(train_size, train_size + test_size)
            ]
            self.attributes["index_list"] = (
                self.attributes["train_index_list"] + self.attributes["test_index_list"]
            )

    def process_data_file(self, file_path, test=False):
        cnt = 0
        with h5py.File(file_path, "r") as h5_file:
            for client_id in tqdm(
                h5_file[self._EXAMPLE].keys(),
                desc="process %s data" % "test" if test else "train",
            ):
                samples = [
                    s.decode("utf8")
                    for s in h5_file[self._EXAMPLE][client_id][self._TOKENS][()]
                ]
                if len(samples) != 0:
                    for sample in samples:
                        idx = len(self.X)
                        self.X[idx] = sample
                        cnt += 1
                        if client_id not in self.nature_partition_dict:
                            self.nature_partition_dict[client_id] = {
                                "train": list(),
                                "test": list(),
                            }
                        if test:
                            self.nature_partition_dict[client_id]["test"].append(idx)
                        else:
                            self.nature_partition_dict[client_id]["train"].append(idx)
        return cnt

    def generate_nature_partition_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        prefix_name = "nature/partition_data/"
        for key in tqdm(
            self.nature_partition_dict.keys(), desc="generate partition h5 file"
        ):
            f[prefix_name + key + "/train"] = self.nature_partition_dict[key]["train"]
            f[prefix_name + key + "/test"] = self.nature_partition_dict[key]["test"]
        f.close()

    @staticmethod
    def generate_partial_h5_file(group_id):
        global global_X
        logging.basicConfig(
            level=logging.INFO,
            format="%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
            datefmt="%Y-%m-%d,%H:%M:%S",
            filename="out_%d.log" % group_id,
            filemode="a",
        )
        f = h5py.File("./tmp_h5_data_group_%d.h5" % group_id, "w")
        start = group_id * CHUNK_SIZE
        end = (
            len(global_X)
            if (group_id + 1) * CHUNK_SIZE > len(global_X)
            else (group_id + 1) * CHUNK_SIZE
        )
        logging.info("child rocess %d start" % group_id)
        for idx in tqdm(range(start, end), desc="group %d loading data" % group_id):
            f["X/" + str(idx)] = global_X[idx]
        f.close()
        logging.info("child process %d end" % group_id)

    def generate_h5_file(self, file_path):
        global global_X
        global_X = self.X
        pool = Pool(15)
        num_groups = math.ceil(len(self.X) / self.CHUNK_SIZE)
        logging.info("total number of groups: %d" % num_groups)
        for i in range(num_groups):
            logging.info("start loading group %d" % i)
            pool.apply_async(self.generate_partial_h5_file, (i,))
        pool.close()
        pool.join()
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        logging.info("start merging")
        for i in tqdm(range(num_groups)):
            tmp_f = h5py.File("./tmp_h5_data_group_%d.h5" % i, "r")
            f.create_group("X_%d" % i)
            tmp_f.copy("X", f["X_%d" % i])
            tmp_f.close()
            os.remove("./tmp_h5_data_group_%d.h5" % i)
        logging.info("done for generating data file")
        f.close()


# if __name__ == "__main__":
#     logging.info("start")
#     data_dir_path = "../../download_scripts/language_model/StackOverflow/datasets/"
#     data_loader = RawDataLoader(data_dir_path)
#     data_loader.load_data()
#     data_loader.generate_h5_file("./stackoverflow_data.h5")
#     data_loader.generate_nature_partition_h5_file("./stackoverflow_partition.h5")
#     logging.info("end")
