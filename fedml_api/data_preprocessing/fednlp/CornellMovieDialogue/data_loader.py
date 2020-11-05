import os
import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *


class DataLoader(BaseDataLoader):
    def __init__(self, data_path, partition, **kwargs):
        super().__init__(data_path, partition, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "history_padding", "source_max_sequence_length",
                        "target_max_sequence_length", "history_max_sequence_length", "vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.history = []
        self.movie_conversation_file_name = "movie_conversations.txt"
        self.movie_line_file_name = "movie_lines.txt"

        if callable(self.partition):
            X, Y, _ = self.process_data(self.data_path)
            self.attributes = self.partition(X, Y)
        else:
            self.attributes = self.process_attributes(self.data_path)

        if self.tokenized:
            self.source_sequence_length = []
            self.target_sequence_length = []
            self.history_sequence_length = []
            self.vocab = dict()
            if self.source_padding or self.target_padding:
                self.vocab[PAD_TOKEN] = len(self.vocab)

            if self.initialize:
                self.vocab[SOS_TOKEN] = len(self.vocab)
                self.vocab[EOS_TOKEN] = len(self.vocab)

    def tokenize(self, document):
        tokens = [str(token) for token in spacy_tokenizer.en_tokenizer(document)]
        return tokens

    def process_attributes(self, file_path):
        attributes = dict()
        attributes["inputs"] = []
        movie_idx_dict = dict()
        with open(os.path.join(file_path, self.movie_conversation_file_name), 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = line.split("+++$+++")
                    conversation_idx = temp[-1].strip()
                    conversation_idx = eval(conversation_idx)
                    if temp[2] not in movie_idx_dict:
                        movie_idx_dict[temp[2]] = len(movie_idx_dict)
                    for i in range(len(conversation_idx) - 1):
                        attributes["inputs"].append(movie_idx_dict[temp[2]])
        attributes["n_clients"] = len(movie_idx_dict)

        return attributes

    def tokenize_data(self, X, Y, history):
        for i in range(len(X)):
            X[i] = self.tokenize(X[i])
            Y[i] = self.tokenize(Y[i])
            history[i] = self.tokenize(history[i])
            self.source_sequence_length.append(len(X[i]))
            self.target_sequence_length.append(len(Y[i]))
            self.history_sequence_length.append(len(history[i]))

    def data_loader(self, client_idx=None):
        X, Y, history = self.process_data(self.data_path, client_idx=client_idx)
        if self.tokenized:
            self.tokenize_data(X, Y, history)
        self.X, self.Y, self.history = X, Y, history
        result = dict()

        if self.tokenized:
            self.build_vocab(self.X, self.vocab)
            self.build_vocab(self.Y, self.vocab)
            result["vocab"] = self.vocab

            if self.source_padding:
                if not self.source_max_sequence_length:
                    self.source_max_sequence_length = max(self.source_sequence_length)
                    if self.initialize:
                        self.source_max_sequence_length += 2
                self.padding_data(self.X, self.source_max_sequence_length, self.initialize)
                result["source_sequence_length"] = self.source_sequence_length
                result["source_max_sequence_length"] = self.source_max_sequence_length
            if self.target_padding:
                if not self.target_max_sequence_length:
                    self.target_max_sequence_length = max(self.target_sequence_length)
                    if self.initialize:
                        self.target_max_sequence_length += 2
                self.padding_data(self.Y, self.target_max_sequence_length, self.initialize)
                result["target_sequence_length"] = self.target_sequence_length
                result["target_max_sequence_length"] = self.target_max_sequence_length
            if self.history_padding:
                if not self.history_max_sequence_length:
                    self.history_max_sequence_length = max(self.history_sequence_length)
                    if self.initialize:
                        self.history_max_sequence_length += 2
                self.padding_data(self.history, self.history_max_sequence_length, self.initialize)
                result["history_sequence_length"] = self.history_sequence_length
                result["history_max_sequence_length"] = self.history_max_sequence_length

        result["attributes"] = self.attributes
        result["X"] = self.X
        result["Y"] = self.Y
        result["history"] = self.history
        return result

    def process_data(self, file_path, client_idx=None):
        line_dict = {}
        with open(os.path.join(file_path, self.movie_line_file_name), "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = line.split("+++$+++")
                    line_dict[temp[0].strip()] = {"utterance": temp[-1].strip()}

        conversation = []
        X = []
        Y = []
        history = []

        cnt = 0
        with open(os.path.join(file_path, self.movie_conversation_file_name), 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = line.split("+++$+++")
                    conversation_idx = temp[-1].strip()
                    conversation_idx = eval(conversation_idx)
                    for i in range(len(conversation_idx) - 1):
                        if client_idx is not None and self.attributes["inputs"][cnt] != client_idx:
                            cnt += 1
                            continue
                        X.append(line_dict[conversation_idx[i]]["utterance"])
                        Y.append(line_dict[conversation_idx[i + 1]]["utterance"])
                        history.append(" ".join(conversation))
                        conversation.append(line_dict[conversation_idx[i]]["utterance"])
                        cnt += 1
                    conversation.clear()
        return X, Y, history

def test_performance():
    import time
    from collections import Counter
    from pympler import asizeof
    train_file_path = "../../../../data/fednlp/seq2seq/CornellMovieDialogue/cornell movie-dialogs corpus/"
    print("uniform partition")
    # load all data
    start = time.time()
    data_loader = DataLoader(train_file_path, uniform_partition, tokenized=True, source_padding=True,
                             target_padding=True, history_padding=True)
    train_data_loader = data_loader.data_loader()
    end = time.time()
    print("all data(tokenized):", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))
    # load a part of data
    start = time.time()
    data_loader = DataLoader(train_file_path, uniform_partition, tokenized=True, source_padding=True,
                             target_padding=True, history_padding=True)
    train_data_loader = data_loader.data_loader(0)
    end = time.time()
    print("part of data(tokenized):", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))

    # load all data
    start = time.time()
    data_loader = DataLoader(train_file_path, uniform_partition)
    train_data_loader = data_loader.data_loader()
    end = time.time()
    print("all data:", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))
    # load a part of data
    start = time.time()
    data_loader = DataLoader(train_file_path, uniform_partition)
    train_data_loader = data_loader.data_loader(0)
    end = time.time()
    print("part of data:", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))

    print("nature partition")
    # load all data
    start = time.time()
    data_loader = DataLoader(train_file_path, "nature", tokenized=True, source_padding=True,
                             target_padding=True, history_padding=True)
    train_data_loader = data_loader.data_loader()
    end = time.time()
    print("all data(tokenized):", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))
    # load a part of data
    attributes = train_data_loader["attributes"]
    total_time = 0
    total_cost = 0
    for client_idx in range(attributes["n_clients"]):
        start = time.time()
        data_loader = DataLoader(train_file_path, "nature", tokenized=True, source_padding=True,
                                 target_padding=True, history_padding=True)

        train_data_loader = data_loader.data_loader(0)
        end = time.time()
        total_time += end - start
        total_cost += asizeof.asizeof(train_data_loader)
    print("part of data(tokenized):", total_time / attributes["n_clients"])
    print("memory cost", total_cost / attributes["n_clients"])
    # load all data
    start = time.time()
    data_loader = DataLoader(train_file_path, uniform_partition)
    train_data_loader = data_loader.data_loader()
    end = time.time()
    print("all data:", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))
    # load a part of data
    attributes = train_data_loader["attributes"]
    total_time = 0
    total_cost = 0
    for client_idx in range(attributes["n_clients"]):
        start = time.time()
        data_loader = DataLoader(train_file_path, "nature", tokenized=True, source_padding=True,
                                 target_padding=True, history_padding=True)

        train_data_loader = data_loader.data_loader(0)
        end = time.time()
        total_time += end - start
        total_cost += asizeof.asizeof(train_data_loader)
    print("part of data:", total_time / attributes["n_clients"])
    print("memory cost", total_cost / attributes["n_clients"])
    print("distribution")
    print(Counter(attributes["inputs"]))


if __name__ == "__main__":
    # train_file_path = "../../../../data/fednlp/seq2seq/CornellMovieDialogue/cornell movie-dialogs corpus/"
    # data_loader = DataLoader(train_file_path, uniform_partition, tokenized=True, source_padding=True, target_padding=True, history_padding=True)
    # train_data_loader = data_loader.data_loader(0)
    # print("done")
    test_performance()
