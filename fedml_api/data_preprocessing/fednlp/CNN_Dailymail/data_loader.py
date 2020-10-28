import struct
from tensorflow.core.example import example_pb2
import sys
sys.path.append('..')
from base.data_loader import BaseDataLoader
from base.globals import *




class DataLoader(BaseDataLoader):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        allowed_keys = {"source_padding", "target_padding", "source_max_sequence_length", "target_max_sequence_length",
                        "source_vocab_path", "target_vocab_path", "initialize"}
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
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

    def data_loader(self):
        self.process_data(self.data_path)

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

        result["X"] = self.X
        result["Y"] = self.Y
        return result

    @staticmethod
    def tokenize(document):
        tokens = []
        for token in document.split(" "):
            token = token.strip()
            if token:
                tokens.append(token)
        return tokens

    @staticmethod
    def process_vocab(vocab_path, vocab):
        with open(vocab_path, "r") as f:
            for line in f:
                line = line.strip()
                token, index = line.split(" ")
                if token not in vocab:
                    vocab[token] = len(vocab)

    def process_data(self, file_path):
        file = open(file_path, "rb")
        while True:
            len_bytes = file.read(8)
            if not len_bytes:
                break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, file.read(str_len))[0]
            example = example_pb2.Example.FromString(example_str)
            article_text = example.features.feature['article'].bytes_list.value[0].decode()
            abstract_text = example.features.feature['abstract'].bytes_list.value[0].decode()
            abstract_text = abstract_text.replace("<s>", "").replace("</s>", "")

            if self.tokenized:
                article_tokens = self.tokenize(article_text)
                abstract_tokens = self.tokenize(abstract_text)

                self.source_sequence_length.append(len(article_tokens))
                self.target_sequence_length.append(len(abstract_tokens))

                self.X.append(article_tokens)
                self.Y.append(abstract_tokens)
            else:
                self.X.append(article_text)
                self.Y.append(abstract_text)



if __name__ == "__main__":
    train_file_path = "../../../../data/fednlp/seq2seq/CNN_Dailymail/finished_files/train.bin"
    dev_file_path = "../../../../data/fednlp/seq2seq/CNN_Dailymail/finished_files/val.bin"
    test_file_path = "../../../../data/fednlp/seq2seq/CNN_Dailymail/finished_files/test.bin"
    vocab_file_path = "../../../../data/fednlp/seq2seq/CNN_Dailymail/finished_files/vocab"
    data_loader = DataLoader(train_file_path, tokenized=True, source_padding=True, target_padding=True)
    train_data_loader = data_loader.data_loader()
    print(train_data_loader["X"][0])
    print(train_data_loader["Y"][0])
    print("done")
