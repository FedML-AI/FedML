from nltk.tokenize import word_tokenize

from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *




class DataLoader(BaseDataLoader):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "history_padding", "source_max_sequence_length",
                        "target_max_sequence_length", "history_max_sequence_length", "vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.history = []
        self.movie_conversation_file_name = "movie_conversations.txt"
        self.movie_line_file_name = "movie_lines.txt"
        self.attributes = dict()

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

    def data_loader(self):
        self.process_data(self.data_path)

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

    def process_data(self, file_path):
        line_dict = {}
        with open(os.path.join(file_path, self.movie_line_file_name), "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = line.split("+++$+++")
                    line_dict[temp[0].strip()] = {"utterance": temp[-1].strip(), "character_id": temp[1]}

        conversation = []
        self.attributes["inputs"] = []

        with open(os.path.join(file_path, self.movie_conversation_file_name), 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = line.split("+++$+++")
                    conversation_idx = temp[-1].strip()
                    conversation_idx = eval(conversation_idx)
                    for i in range(len(conversation_idx) - 1):
                        if self.tokenized:
                            tokens = self.tokenize(line_dict[conversation_idx[i]]["utterance"])
                            next_tokens = self.tokenize(line_dict[conversation_idx[i+1]]["utterance"])
                            self.X.append(tokens)
                            self.Y.append(next_tokens)
                            self.history.append(conversation.copy())
                            self.source_sequence_length.append(len(tokens))
                            self.target_sequence_length.append(len(next_tokens))
                            self.history_sequence_length.append(len(conversation))
                            conversation += tokens
                        else:
                            self.X.append(line_dict[conversation_idx[i]]["utterance"])
                            self.Y.append(line_dict[conversation_idx[i + 1]]["utterance"])
                            self.history.append(" ".join(conversation))
                            conversation.append(line_dict[conversation_idx[i]]["utterance"])

                        character_id = line_dict[conversation_idx[i]]["character_id"]
                        next_character_id = line_dict[conversation_idx[i]]["character_id"]

                        self.attributes["inputs"].append(
                            {"character_id": character_id, "next_character_id": next_character_id,
                             "movie_id": temp[2]})
                    conversation.clear()


    @staticmethod
    def partition(keys, values, attributes):
        movie_dict = dict()
        for attribute in attributes["inputs"]:
            if attribute["movie_id"] not in movie_dict:
                movie_dict[attribute["movie_id"]] = len(movie_dict)
        length = len(values[0])
        result = dict()
        for key in keys:
            result[key] = dict()
        for i in range(length):
            client_idx = movie_dict[attributes["inputs"][i]["movie_id"]]
            for j, key in enumerate(keys):
                if client_idx not in result[key]:
                    result[key][client_idx] = [values[j][i]]
                else:
                    result[key][client_idx].append(values[j][i])
                    result[key][client_idx].append(values[j][i])
        return result


if __name__ == "__main__":
    import os
    import sys
    sys.path.append('..')
    train_file_path = "../../../../data/fednlp/seq2seq/CornellMovieDialogue/CornellMovieDialogue/"
    data_loader = DataLoader(train_file_path, tokenized=True, source_padding=True, target_padding=True, history_padding=True)
    train_data_loader = data_loader.data_loader()
    partition(train_data_loader, method="uniform")
    print("done")
