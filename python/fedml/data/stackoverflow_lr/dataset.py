import h5py

import torch.utils.data as data


class StackOverflowDataset(data.Dataset):
    """StackOverflow dataset"""

    __train_client_id_list = None
    __test_client_id_list = None

    def __init__(self, h5_path, client_idx, datast, preprocess=None):
        """
        Args:
            h5_path (string) : path to the h5 file
            client_idx (idx) : index of train file
            datast (string) : "train" or "test" denoting on train set or test set
            preprocess (dict of callable, optional) : Optional preprocessing, with key "input", "target"
        """

        self._EXAMPLE = "examples"
        self._TOKENS = "tokens"
        self._TITLE = "title"
        self._TAGS = "tags"

        self.h5_path = h5_path
        self.datast = datast
        self.client_id = self.get_client_id_list()[client_idx]  # pylint: disable=E1136
        self.input_fn = None
        self.target_fn = None
        if preprocess:
            self.input_fn = preprocess["input"]
            self.target_fn = preprocess["target"]

    def get_client_id_list(self):
        if self.datast == "train":
            if StackOverflowDataset.__train_client_id_list is None:
                with h5py.File(self.h5_path, "r") as h5_file:
                    StackOverflowDataset.__train_client_id_list = list(
                        h5_file[self._EXAMPLE].keys()
                    )
            return StackOverflowDataset.__train_client_id_list
        elif self.datast == "test":
            if StackOverflowDataset.__test_client_id_list is None:
                with h5py.File(self.h5_path, "r") as h5_file:
                    StackOverflowDataset.__test_client_id_list = list(
                        h5_file[self._EXAMPLE].keys()
                    )
            return StackOverflowDataset.__test_client_id_list
        else:
            raise Exception("Please specify either train or test set!")

    def __len__(self):
        with h5py.File(self.h5_path, "r") as h5_file:
            return len(h5_file[self._EXAMPLE][self.client_id][self._TAGS][()])

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as h5_file:
            if idx > self.__len__():
                return None
            raw_token = h5_file[self._EXAMPLE][self.client_id][self._TOKENS][()][
                idx
            ].decode("utf8")
            raw_title = h5_file[self._EXAMPLE][self.client_id][self._TITLE][()][
                idx
            ].decode("utf8")
            sample = " ".join([raw_token, raw_title])
            tag = h5_file[self._EXAMPLE][self.client_id][self._TAGS][()][idx].decode(
                "utf8"
            )

            if self.input_fn:
                sample = self.input_fn(sample)
            if self.target_fn:
                tag = self.target_fn(tag)

        return (sample, tag)
