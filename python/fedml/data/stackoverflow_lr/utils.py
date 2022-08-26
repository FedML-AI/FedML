import collections
import json
import os

import numpy as np

DEFAULT_WORD_COUNT_FILE = "stackoverflow.word_count"
DEFAULT_TAG_COUNT_FILE = "stackoverflow.tag_count"
word_count_file_path = None
tag_count_file_path = None
word_dict = None
tag_dict = None
"""
This code follows the steps of preprocessing in tff stackoverflow dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/stackoverflow_lr_dataset.py
"""


def get_word_count_file(data_dir):
    # word_count_file_path
    global word_count_file_path
    if word_count_file_path is None:
        word_count_file_path = os.path.join(data_dir, DEFAULT_WORD_COUNT_FILE)
    return word_count_file_path


def get_tag_count_file(data_dir):
    # tag_count_file_path
    global tag_count_file_path
    if tag_count_file_path is None:
        tag_count_file_path = os.path.join(data_dir, DEFAULT_TAG_COUNT_FILE)
    return tag_count_file_path


def get_most_frequent_words(data_dir=None, vocab_size=10000):
    frequent_words = []
    with open(get_word_count_file(data_dir), "r") as f:
        frequent_words = [next(f).split()[0] for i in range(vocab_size)]
    return frequent_words


def get_tags(data_dir=None, tag_size=500):
    f = open(get_tag_count_file(data_dir), "r")
    frequent_tags = json.load(f)
    return list(frequent_tags.keys())[:tag_size]


def get_word_dict(data_dir):
    global word_dict
    if word_dict == None:
        words = get_most_frequent_words(data_dir)
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict


def get_tag_dict(data_dir):
    global tag_dict
    if tag_dict == None:
        tags = get_tags(data_dir)
        tag_dict = collections.OrderedDict()
        for i, w in enumerate(tags):
            tag_dict[w] = i
    return tag_dict


def preprocess_inputs(sentences, data_dir):

    sentences = [sentence.split(" ") for sentence in sentences]
    vocab_size = len(get_word_dict(data_dir))

    def word_to_id(word):
        word_dict = get_word_dict(data_dir)
        if word in word_dict:
            return word_dict[word]
        else:
            return len(word_dict)

    def to_bag_of_words(sentence):
        tokens = [word_to_id(token) for token in sentence]
        onehot = np.zeros((len(tokens), vocab_size + 1))
        onehot[np.arange(len(tokens)), tokens] = 1
        return np.mean(onehot, axis=0)[:vocab_size]

    return [to_bag_of_words(sentence) for sentence in sentences]


def preprocess_targets(tags, data_dir):

    tags = [tag.split("|") for tag in tags]
    tag_size = len(get_tag_dict(data_dir))

    def tag_to_id(tag):
        tag_dict = get_tag_dict(data_dir)
        if tag in tag_dict:
            return tag_dict[tag]
        else:
            return len(tag_dict)

    def to_bag_of_words(tag):
        tag = [tag_to_id(t) for t in tag]
        onehot = np.zeros((len(tag), tag_size + 1))
        onehot[np.arange(len(tag)), tag] = 1
        return np.sum(onehot, axis=0, dtype=np.float32)  # [:tag_size]

    return [to_bag_of_words(tag) for tag in tags]


def preprocess_input(sentence, data_dir):

    sentence = sentence.split(" ")
    vocab_size = len(get_word_dict(data_dir))

    def word_to_id(word):
        word_dict = get_word_dict(data_dir)
        if word in word_dict:
            return word_dict[word]
        else:
            return len(word_dict)

    def to_bag_of_words(sentence):
        tokens = [word_to_id(token) for token in sentence]
        onehot = np.zeros((len(tokens), vocab_size + 1))
        onehot[np.arange(len(tokens)), tokens] = 1
        return np.mean(onehot, axis=0, dtype=np.float32)[:vocab_size]

    return to_bag_of_words(sentence)


def preprocess_target(tag, data_dir):

    tag = tag.split("|")
    tag_size = len(get_tag_dict(data_dir))

    def tag_to_id(tag):
        tag_dict = get_tag_dict(data_dir)
        if tag in tag_dict:
            return tag_dict[tag]
        else:
            return len(tag_dict)

    def to_bag_of_words(tag):
        tag = [tag_to_id(t) for t in tag]
        onehot = np.zeros((len(tag), tag_size + 1))
        onehot[np.arange(len(tag)), tag] = 1
        return np.sum(onehot, axis=0, dtype=np.float32)[:tag_size]

    return to_bag_of_words(tag)
