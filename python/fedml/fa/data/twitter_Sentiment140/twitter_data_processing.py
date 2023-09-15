"""Preprocess twitter data
https://github.com/triehh/triehh/blob/master/preprocess.py
"""
import collections
import csv
import operator
import os
import pickle
import re


def is_valid(word):
    """
    Check if a word is valid for processing.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is valid, False otherwise.
    """
    if len(word) < 3 or (word[-1] in ['?', '!', '.', ';', ',']) or word.startswith('http') or word.startswith('www'):
        return False
    if re.match(r'^[a-z_\@\#\-\;\(\)\*\:\.\'\/]+$', word):
        return True
    return False


def truncate_or_extend(word, max_word_len):
    """
    Truncate or extend a word to a specified length.

    Args:
        word (str): The word to modify.
        max_word_len (int): The desired maximum length of the word.

    Returns:
        str: The modified word.
    """
    if len(word) > max_word_len:
        word = word[:max_word_len]
    else:
        word += '$' * (max_word_len - len(word))
    return word


def add_end_symbol(word):
    """
    Add an end symbol ('$') to a word.

    Args:
        word (str): The word to modify.

    Returns:
        str: The modified word with an end symbol.
    """
    return word + '$'


def generate_triehh_clients(clients, path):
    """
    Generate TrieHH clients from a list of clients and save them to a file.

    Args:
        clients (list): List of client names.
        path (str): The directory path to save the file.
    """
    clients_num = len(clients)
    triehh_clients = [add_end_symbol(clients[i]) for i in range(clients_num)]
    word_freq = collections.defaultdict(lambda: 0)
    for word in triehh_clients:
        word_freq[word] += 1
    word_freq = dict(word_freq)
    file_path = os.path.join(path, 'clients_triehh.txt')
    with open(file_path, 'wb') as fp:
        pickle.dump(triehh_clients, fp)


def preprocess_twitter_data(path):
    """
    Preprocess Twitter data from a CSV file.

    Args:
        path (str): The directory path where the CSV file is located.

    Returns:
        dict: A dictionary containing client usernames as keys and lists of preprocessed words as values.
    """
    filename = os.path.join(path, 'training.1600000.processed.noemoticon.csv')
    dataset = {}
    with open(filename, encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 6:
                client = row[4]
                comment = row[5]

                raw_words = comment.lower().split()
                raw_words = [word.strip(',.;?!') for word in raw_words]
                words = [x for x in raw_words if is_valid(x)]

                word_len = len(words)
                if word_len > 0:
                    if client not in dataset:
                        dataset[client] = list()
                    dataset[client].extend(words)
    return dataset


def preprocess_twitter_data_heavy_hitter(path):
    """
    Preprocess Twitter data and identify heavy hitters (most frequent words) for each client.

    Args:
        path (str): The directory path where the CSV file is located.

    Returns:
        dict: A dictionary containing client usernames as keys and their identified heavy hitter words as values.
    """
    # load dataset from csv file
    filename = os.path.join(path, 'training.1600000.processed.noemoticon.csv')
    clients = {}
    with open(filename, encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 6:
                client = row[4]
                comment = row[5]

                raw_words = comment.lower().split()
                raw_words = [word.strip(',.;?!') for word in raw_words]
                words = [x for x in raw_words if is_valid(x)]

                # don't create client if he/she has no valid words
                word_len = len(words)
                if word_len > 0 and client not in clients:
                    clients[client] = {}
                for word in words:
                    if word not in clients[client]:
                        clients[client][word] = 1
                    else:
                        clients[client][word] += 1

    # get the top word for every client
    local_datasets_top_word = {}
    for client in clients:
        top_word = max(clients[client].items(), key=operator.itemgetter(1))[0]
        local_datasets_top_word[client] = add_end_symbol(top_word)
    return local_datasets_top_word