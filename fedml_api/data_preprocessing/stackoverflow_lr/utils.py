import numpy as np
import json
import collections

word_count_file_path = '../../../data/stackoverflow/datasets/stackoverflow.word_count'
tag_count_file_path = '../../../data/stackoverflow/datasets/stackoverflow.tag_count'
word_dict = None
tag_dict = None
'''
This code follows the steps of preprocessing in tff stackoverflow dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/stackoverflow_lr_dataset.py
'''


def get_most_frequent_words(vocab_size=10000):
    frequent_words = []
    with open(word_count_file_path, 'r') as f:
        frequent_words = [next(f).split()[0] for i in range(vocab_size)]
    return frequent_words


def get_tags(tag_size=500):
    f = open(tag_count_file_path, 'r')
    frequent_tags = json.load(f)
    return list(frequent_tags.keys())[:tag_size]


def get_word_dict():
    global word_dict
    if word_dict == None:
        words = get_most_frequent_words()
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict


def get_tag_dict():
    global tag_dict
    if tag_dict == None:
        tags = get_tags()
        tag_dict = collections.OrderedDict()
        for i, w in enumerate(tags):
            tag_dict[w] = i
    return tag_dict


def word_to_id(word):
    word_dict = get_word_dict()
    if word in word_dict:
        return word_dict[word]
    else:
        return len(word_dict)


def tag_to_id(tag):
    tag_dict = get_tag_dict()
    if tag in tag_dict:
        return tag_dict[tag]
    else:
        return len(tag_dict)


def preprocess_inputs(sentences):

    sentences = [sentence.split(' ') for sentence in sentences]
    vocab_size = len(get_word_dict())

    def to_bag_of_words(sentence):
        tokens = [word_to_id(token) for token in sentence]
        onehot = np.zeros((len(tokens), vocab_size + 1))
        onehot[np.arange(len(tokens)), tokens] = 1
        return np.mean(onehot, axis=0)[:vocab_size]

    return [to_bag_of_words(sentence) for sentence in sentences]


def preprocess_targets(tags):

    tags = [tag.split('|') for tag in tags]
    tag_size = len(get_tag_dict())

    def to_bag_of_words(tag):
        tag = [tag_to_id(t) for t in tag]
        onehot = np.zeros((len(tag), tag_size + 1))
        onehot[np.arange(len(tag)), tag] = 1
        return np.sum(onehot, axis=0)#[:tag_size]

    return [to_bag_of_words(tag) for tag in tags]


if __name__ == "__main__":
    inputs = [
        'this will output :',
        'the simplest way i know how to do that is to move the file , delete the file using svn , and then move the file back .',
    ]
    processed_inputs = preprocess_inputs(inputs)
    print(np.shape(processed_inputs))
    print(processed_inputs)

    targets = [
        'asp . net|flash|voice-recording',
        'jquery|google-chrome|greasemonkey|require|userscripts',
        'sql-server|indexing'
    ]
    processed_targets = preprocess_targets(targets)
    print(np.shape(processed_targets))
    print(processed_targets)
