import os
import sys
from pathlib import Path
import logging
import datetime

import pickle



def pickle_save(data_obj, file_path):
    with open(file_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data_obj, f)


def pickle_load(file_path):
    if file_path.exists() and os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as f:
            loaded_obj = pickle.load(f)
    else:
        loaded_obj = {}
    return loaded_obj





class FedMLLocalCache:

    cache_name = "fedml_client"
    cache_dir = "fedml_client_caches"

    @classmethod
    def init(cls, args, root):
        # path = os.path.join(root, FedMLLocalCache.cache_dir)
        # if os.path.exists(path):
        #     os.makedirs(path)
        TIMEFORMAT = '%Y%m%d_%H%M%S'
        theTime = datetime.datetime.now()
        str_time = theTime.strftime(TIMEFORMAT) + theTime.strftime('%f')[:2]
        # print(theTime)
        # print(str_time)
        cls.path = Path(root) / FedMLLocalCache.cache_dir / f"fedml_{str_time}"
        if not cls.path.exists():
            cls.path.mkdir(parents=True)
        # else:
        #     # Clear the cache of last training.
        #     cls.path.unlink()

    # @classmethod
    # def save(cls, args, root, client_index, key, value):
    #     # dir_name, file_name = FedMLLocalCache.get_file_name(args, root, client_index)
    #     # with open(file_name, 'wb') as f:
    #     #     pickle.dump(some_obj, f)
    #     file_path = cls.path / FedMLLocalCache.cache_name + f"_{client_index}"
    #     if file_path.exists():
    #         with open(file_path, 'rb') as f:
    #             loaded_obj = pickle.load(f)
    #     else:
    #         loaded_obj = {}
    #     loaded_obj[key] = value
    #     pickle_save(loaded_obj, file_path)


    @classmethod
    def save(cls, args, client_index, save_obj):
        file_path = cls.path / f"{FedMLLocalCache.cache_name}_{client_index}"
        logging.info(f"save obj, file_path: {file_path}")
        pickle_save(save_obj, file_path)

    @classmethod
    def load(cls, args, client_index):
        file_path = cls.path / f"{FedMLLocalCache.cache_name}_{client_index}"
        logging.info(f"load obj, file_path: {file_path}")
        # with open(file_path, 'rb') as f:
        #     loaded_obj = pickle.load(f)
        loaded_obj = pickle_load(file_path)
        return loaded_obj


    # @classmethod
    # def get_file_name(cls, args, root, client_index):
    #     FedMLLocalCache.cache_name
    #     dir_name = os.path.join(root, FedMLLocalCache.cache_dir)
    #     file_name = os.path.join(dir_name, FedMLLocalCache.cache_name + f"_{client_index}")
    #     file_name = file_name + ".pt"
    #     return dir_name, file_name


    @classmethod
    def finalize(cls):
        pass












