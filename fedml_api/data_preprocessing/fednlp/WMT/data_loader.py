import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.globals import *
from base.partition import *
import time



language_pairs = [("cs", "en"), ("de", "en"), ("ru", "en"), ("zh", "en")]


class DataLoader(BaseDataLoader):
    def __init__(self, data_path, partition, **kwargs):
        super().__init__(data_path, partition, **kwargs)
        allowed_keys = {"language_pair", "source_padding", "target_padding", "source_max_sequence_length",
                        "target_max_sequence_length", "source_vocab_path", "target_vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)

        X, Y = self.process_data(self.data_path)
        self.attributes = self.partition(X, Y)

        if self.tokenized:
            self.source_sequence_length = []
            self.target_sequence_length = []
            self.source_vocab = dict()
            self.target_vocab = dict()
            if self.source_padding:
                self.source_vocab[PAD_TOKEN] = len(self.source_vocab)
            if self.target_padding:
                self.target_vocab[PAD_TOKEN] = len(self.target_vocab)
            if self.initialize:
                self.source_vocab[SOS_TOKEN] = len(self.source_vocab)
                self.source_vocab[EOS_TOKEN] = len(self.source_vocab)
                self.target_vocab[SOS_TOKEN] = len(self.target_vocab)
                self.target_vocab[EOS_TOKEN] = len(self.target_vocab)

    def tokenize_data(self, X, Y, source_lang, target_lang):
        for i in range(len(X)):
            X[i] = self.tokenize(X[i], source_lang)
            Y[i] = self.tokenize(Y[i], target_lang)
            self.source_sequence_length.append(len(X[i]))
            self.target_sequence_length.append(len(Y[i]))

    def data_loader(self, client_idx=None):
        X, Y = self.process_data(self.data_path, client_idx)
        if self.tokenized:
            source_lang = self.data_path[0].split(".")[-1]
            target_lang = self.data_path[1].split(".")[-1]
            self.tokenize_data(X, Y, source_lang, target_lang)
        self.X, self.Y = X, Y
        result = dict()

        if self.tokenized:
            if self.source_vocab_path:
                self.process_vocab(self.source_vocab_path, self.source_vocab)
            else:
                self.build_vocab(self.X, self.source_vocab)
            result["source_vocab"] = self.source_vocab

            if self.target_vocab_path:
                self.process_vocab(self.target_vocab_path, self.target_vocab)
            else:
                self.build_vocab(self.Y, self.target_vocab)
            result["target_vocab"] = self.target_vocab

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
        result["attributes"] = self.attributes
        result["X"] = self.X
        result["Y"] = self.Y
        return result

    @staticmethod
    def tokenize(document, lang):
        tokenizer = None
        if lang == "zh":
            tokenizer = spacy_tokenizer.zh_tokenizer
        elif lang == "en":
            tokenizer = spacy_tokenizer.en_tokenizer
        elif lang == "cs":
            tokenizer = spacy_tokenizer.cs_tokenizer
        elif lang == "de":
            tokenizer = spacy_tokenizer.de_tokenizer
        elif lang == "ru":
            tokenizer = spacy_tokenizer.ru_tokenizer
        else:
            raise Exception("Unacceptable language.")
        tokens = [str(token) for token in tokenizer(document)]
        return tokens

    def process_data(self, file_path, client_idx=None):
        source_path = file_path[0]
        target_path = file_path[1]
        X = []
        Y = []
        cnt = 0
        with open(source_path, "r") as f:
            for line in f:
                if client_idx is not None and client_idx != self.attributes["inputs"][cnt]:
                    cnt += 1
                    continue
                line = line.strip()
                X.append(line)
                cnt += 1
        cnt = 0
        with open(target_path, "r") as f:
            for line in f:
                if client_idx is not None and client_idx != self.attributes["inputs"][cnt]:
                    cnt += 1
                    continue
                line = line.strip()
                Y.append(line)
                cnt += 1
        return X, Y

def test_performance():
    import time
    from pympler import asizeof
    train_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.cs",
                        "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.en"]
    # load all data
    start = time.time()
    data_loader = DataLoader(train_file_paths, uniform_partition, tokenized=True, source_padding=True,
                             target_padding=True)
    train_data_loader = data_loader.data_loader()
    end = time.time()
    print("all data(tokenized):", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))
    # load a part of data
    start = time.time()
    data_loader = DataLoader(train_file_paths, uniform_partition, tokenized=True, source_padding=True,
                             target_padding=True)
    train_data_loader = data_loader.data_loader(0)
    end = time.time()
    print("part of data(tokenized):", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))

    # load all data
    start = time.time()
    data_loader = DataLoader(train_file_paths, uniform_partition)
    train_data_loader = data_loader.data_loader()
    end = time.time()
    print("all data:", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))
    # load a part of data
    start = time.time()
    data_loader = DataLoader(train_file_paths, uniform_partition)
    train_data_loader = data_loader.data_loader(0)
    end = time.time()
    print("part of data:", end - start)
    print("size", len(train_data_loader["X"]))
    print("memory cost", asizeof.asizeof(train_data_loader))

if __name__ == "__main__":
    # train_file_paths = ["../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.cs",
    #                     "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.cs-en.en"]
    # data_loader = DataLoader(train_file_paths, uniform_partition, tokenized=True, source_padding=True,
    #                          target_padding=True)
    # train_data_loader = data_loader.data_loader(0)
    # print("done")
    test_performance()
