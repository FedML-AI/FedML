import collections
import os

import numpy as np

DEFAULT_WORD_COUNT_FILE = "stackoverflow.word_count"
word_count_file_path = None
word_dict = None
word_list = None
_pad = "<pad>"
_bos = "<bos>"
_eos = "<eos>"
"""
This code follows the steps of preprocessing in tff stackoverflow dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/stackoverflow_dataset.py
"""


def get_word_count_file(data_dir):
    # word_count_file_path
    global word_count_file_path
    if word_count_file_path is None:
        word_count_file_path = os.path.join(data_dir, DEFAULT_WORD_COUNT_FILE)
    return word_count_file_path


def get_most_frequent_words(data_dir, vocab_size=10000):
    frequent_words = []
    with open(get_word_count_file(data_dir), "r") as f:
        frequent_words = [next(f).split()[0] for i in range(vocab_size)]
    return frequent_words


def get_word_dict(data_dir):
    global word_dict
    if word_dict == None:
        frequent_words = get_most_frequent_words(data_dir)
        words = [_pad] + frequent_words + [_bos] + [_eos]
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict


def get_word_list(data_dir):
    global word_list
    if word_list == None:
        word_dict = get_word_dict(data_dir)
        word_list = list(word_dict.keys())
    return word_list


def tokenizer(sentence, data_dir, max_seq_len=20):

    truncated_sentences = sentence.split(" ")[:max_seq_len]

    def word_to_id(word, num_oov_buckets=1):
        word_dict = get_word_dict(data_dir)
        if word in word_dict:
            return word_dict[word]
        else:
            return hash(word) % num_oov_buckets + len(word_dict)

    def to_ids(sentence, num_oov_buckets=1):
        """
        map list of sentence to list of [idx..] and pad to max_seq_len + 1
        Args:
            num_oov_buckets : The number of out of vocabulary buckets.
            max_seq_len: Integer determining shape of padded batches.
        """
        tokens = [word_to_id(token) for token in sentence]
        if len(tokens) < max_seq_len:
            tokens = tokens + [word_to_id(_eos)]
        tokens = [word_to_id(_bos)] + tokens
        if len(tokens) < max_seq_len + 1:
            tokens += [word_to_id(_pad)] * (max_seq_len + 1 - len(tokens))
        return tokens

    return to_ids(truncated_sentences)


def split(dataset):
    ds = np.array(dataset)
    x = ds[:, :-1]
    y = ds[:, -1]
    return x, y
