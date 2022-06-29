from data.data_manager.base_data_manager import BaseDataManager
from torch.utils.data import DataLoader
import h5py
import json
import logging
from tqdm import tqdm


class LanguageModelDataManager(BaseDataManager):
    """Data manager for language model tasks."""

    def __init__(self, args, model_args, preprocessor, process_id=0, num_workers=1):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)

        super(LanguageModelDataManager, self).__init__(
            args, model_args, process_id, num_workers
        )
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

    def read_instance_from_h5(self, data_file, index_list, split = "", desc=""):
        X = list()
        for idx in tqdm(index_list, desc="Loading data from h5 file." + desc):
            X.append(data_file["X"][str(idx)][()].decode("utf-8"))
        return {"X": X}


class StackOverFlowDataManager(LanguageModelDataManager):
    """Data manager for stackoverflow tasks."""

    def __init__(self, args, model_args, preprocessor, process_id=0, num_workers=1):
        super(StackOverFlowDataManager, self).__init__(
            args, model_args, preprocessor, process_id, num_workers
        )
        self.CHUNK_SIZE = 1000000

    def read_instance_from_h5(self, data_file, index_list, desc=""):
        X = list()
        for idx in tqdm(index_list, desc="Loading data from h5 file." + desc):
            group_id = int(idx / self.CHUNK_SIZE)
            X.append(data_file["X_%d/X" % group_id][str(idx)][()].decode("utf-8"))
        return {"X": X}
