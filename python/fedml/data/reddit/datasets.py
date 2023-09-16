from __future__ import print_function

import glob
import json
import os
import os.path
import pickle
import time
import warnings

import torch
import torch.nn.functional as F


class Reddit_dataset():
    """
    Dataset class for Reddit data.

    Args:
        root (str): The root directory where the data is stored.
        train (bool): Whether to load the training or testing dataset.

    Attributes:
        train_file (str): The file name for the training dataset.
        test_file (str): The file name for the testing dataset.
        vocab_tokens_size (int): The size of the token vocabulary.
        vocab_tags_size (int): The size of the tag vocabulary.
        raw_data (list): A list of tokenized text data.
        dict (dict): A mapping dictionary from sample id to target tag.

    Methods:
        __getitem__(self, index):
            Get an item from the dataset by index.
        __mapping_dict__(self):
            Get the mapping dictionary.
        __len__(self):
            Get the length of the dataset.
        raw_folder(self):
            Get the raw data folder path.
        processed_folder(self):
            Get the processed data folder path.
        class_to_idx(self):
            Get a mapping from class names to class indices.
        _check_exists(self):
            Check if the dataset exists.
        load_token_vocab(self, vocab_size, path):
            Load token vocabulary from a file.
        load_file(self, path, is_train):
            Load the dataset from files.

    """
    classes = []
    MAX_SEQ_LEN = 20000

    def __init__(self, root, train=True):
        """
        Initialize the Reddit_dataset.

        Args:
            root (str): The root directory where the data is stored.
            train (bool): Whether to load the training or testing dataset.
        """
        self.train = train  # training set or test set
        self.root = root

        self.train_file = 'train'
        self.test_file = 'test'
        self.train = train

        self.vocab_tokens_size = 10000
        self.vocab_tags_size = 500

        # load data and targets
        self.raw_data, self.dict = self.load_file(self.root, self.train)

        if not self.train:
            self.raw_data = self.raw_data[:100000]
        else:
            self.raw_data = self.raw_data[:10000000]

        # we can't enumerate the raw data, thus generating artificial data to cheat the divide_data_loader
        self.data = [-1, len(self.dict)]
        self.targets = [-1, len(self.dict)]

    def __getitem__(self, index):
        """
        Args:xx
            index (int): Index
        Returns:
            tuple: (text, tags)
        """

        # Lookup tensor

        tokens = self.raw_data[index]
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = F.one_hot(tokens, self.vocab_tokens_size).float()
        tokens = tokens.mean(0)

        # tags = torch.tensor(tags, dtype=torch.long)
        # tags = F.one_hot(tags, self.vocab_tags_size).float()
        # tags = tags.sum(0)

        return tokens

    def __mapping_dict__(self):
        """
        Get the mapping dictionary.

        Returns:
            dict: A dictionary mapping sample IDs to target tags.
        """

        return self.dict

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.raw_data)

    @property
    def raw_folder(self):
        """
        Get the raw data folder path.

        Returns:
            str: The path to the raw data folder.
        """
        return self.root

    @property
    def processed_folder(self):
        """
        Get the processed data folder path.

        Returns:
            str: The path to the processed data folder.
        """
        return self.root

    @property
    def class_to_idx(self):
        """
        Get a mapping from class names to class indices.

        Returns:
            dict: A dictionary mapping class names to class indices.
        """
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        """
        Check if the dataset exists.

        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        return (os.path.exists(os.path.join(self.processed_folder, self.data_file)))

    def load_token_vocab(self, vocab_size, path):
        """
        Load token vocabulary from a file.

        Args:
            vocab_size (int): The size of the token vocabulary.
            path (str): The path to the vocabulary file.

        Returns:
            list: A list of tokens from the vocabulary.
        """
        tokens_file = "reddit_vocab.pkl"
        with open(os.path.join(path, tokens_file), 'rb') as f:
            tokens = pickle.load(f)
        return tokens[:vocab_size]

    def load_file(self, path, is_train):
        """
        Load the dataset from files.

        Args:
            path (str): The path to the dataset files.
            is_train (bool): Whether to load the training or testing dataset.

        Returns:
            tuple: A tuple containing text data and a mapping dictionary.
        """
        file_name = os.path.join(
            path, 'train') if self.train else os.path.join(path, 'test')

        # check whether we have generated the cache file before
        cache_path = os.path.join(
            path, "train_cache") if self.train else os.path.join(path, "test_cache")

        text = []
        mapping_dict = {}

        if os.path.exists(cache_path):
            print("====Load {} from cache".format(file_name))
            # dump the cache
            with open(cache_path, 'rb') as fin:
                text = pickle.load(fin)
                mapping_dict = pickle.load(fin)
        else:
            print("====Load {} from scratch".format(file_name))
            # Mapping from sample id to target tag
            # First, get the token and tag dict
            vocab_tokens = self.load_token_vocab(self.vocab_tokens_size, path)

            vocab_tokens_dict = {k: v for v, k in enumerate(vocab_tokens)}
            client_data_list = []
            # Load the traning/testing data
            if self.train:
                train_files = sorted(glob.glob(file_name + "/*.json"))

                for f in train_files[:2]:
                    print("========Loading {}=========".format(f))
                    with open(f, 'rb') as cin:
                        data = json.load(cin)
                    client_data_list.append(data)

            else:
                with open(os.path.join(file_name, "test.json"), 'rb') as cin:
                    data = json.load(cin)
                client_data_list.append(data)

            count = 0
            clientCount = 0
            start_time = time.time()

            for client_data in client_data_list:
                client_list = client_data['users']

                for clientId, client in enumerate(client_list):
                    tokens_list = list(client_data['user_data'][client]['x'])

                    for tokens in tokens_list:

                        tokens_list = [vocab_tokens_dict[s] for s in (
                            tokens.split()) if s in vocab_tokens_dict]
                        if not tokens_list:
                            continue

                        mapping_dict[count] = clientId
                        text.append(tokens_list)

                        count += 1

                    clientCount += 1

                    # num_of_remains = 1628176 - int(client)
                    #print("====In loading data, remains {} clients, may take {} sec".format(num_of_remains, (time.time() - start_time)/clientCount * num_of_remains))
                    # logging.info("====In loading  data, remains {} clients".format(num_of_remains)

                    if clientId % 5000 == 0:
                        # dump the cache
                        with open(cache_path, 'wb') as fout:
                            pickle.dump(text, fout)
                            pickle.dump(mapping_dict, fout)

                        print("====Dump for {} clients".format(clientId))

            # dump the cache
            with open(cache_path, 'wb') as fout:
                pickle.dump(text, fout)
                pickle.dump(mapping_dict, fout)

        return text, mapping_dict
