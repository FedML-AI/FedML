from ..data_manager.base_data_manager import BaseDataManager
from torch.utils.data import DataLoader
import h5py
import json
import logging
from tqdm import tqdm


class TextClassificationDataManager(BaseDataManager):
    """Data manager for text classification"""

    def __init__(self, args, model_args, preprocessor, process_id=0, num_workers=1):
        # TODO: ref to a defination of the "args" and "model_args"
        #           --- what will they must contain? (e.g., data_file_path)

        super(TextClassificationDataManager, self).__init__(
            args, model_args, process_id, num_workers
        )
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

    def read_instance_from_h5(self, data_file, index_list, split = "", desc=""):
        X = list()
        y = list()
        for idx in tqdm(index_list, desc="Loading data from h5 file." + desc):
            if self.args.dataset == "20news" and split == "test" and self.args.partition_method == "niid_label_clients=100_alpha=1.0":
                X.append(data_file["X"][str(idx + 11314)][()].decode("utf-8"))
            else:
                X.append(data_file["X"][str(idx)][()].decode("utf-8"))
            if self.args.dataset == "20news" and split == "test" and self.args.partition_method == "niid_label_clients=100_alpha=1.0":
                y.append(data_file["Y"][str(idx + 11314)][()].decode("utf-8"))
            else:
                y.append(data_file["Y"][str(idx)][()].decode("utf-8"))
        return {"X": X, "y": y}
