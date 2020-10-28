from base.data_loader import BaseDataLoader
from base.globals import *



language_pairs = [("cs", "en"), ("de", "en"), ("ru", "en"), ("zh", "en")]


class DataLoader(BaseDataLoader):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path, **kwargs)
        allowed_keys = {"language_pair", "source_padding", "target_padding", "source_max_sequence_length",
                        "target_max_sequence_length", "source_vocab_path", "target_vocab_path", "initialize"}
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
        self.process_data(self.data_path.format(self.language_pair[0], self.language_pair[1], self.language_pair[0]),
                          self.language_pair[0], True)
        self.process_data(self.data_path.format(self.language_pair[0], self.language_pair[1], self.language_pair[1]),
                          self.language_pair[1], False)

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
    def tokenize(document, lang):
        tokenizer = None
        if lang == "zh":
            tokenizer = zh_tokenizer
        elif lang == "en":
            tokenizer = en_tokenizer
        elif lang == "cs":
            tokenizer = cs_tokenizer
        elif lang == "de":
            tokenizer = de_tokenizer
        elif lang == "ru":
            tokenizer = ru_tokenizer
        else:
            raise Exception("Unacceptable language.")
        tokens = [str(token) for token in tokenizer(document)]
        return tokens

    def process_data(self, file_path, lang, source):
        if source:
            X = self.X
            sequence_length = self.source_sequence_length if self.tokenized else None
        else:
            X = self.Y
            sequence_length = self.target_sequence_length if self.tokenized else None

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if self.tokenized:
                    tokens = self.tokenize(line, lang)
                    X.append(tokens)
                    sequence_length.append(len(tokens))
                else:
                    X.append(line)


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    train_file_path = "../../../../data/fednlp/seq2seq/WMT/training-parallel-nc-v13/news-commentary-v13.{}-{}.{}"
    data_loader = DataLoader(train_file_path, language_pair=("cs", "en"), tokenized=True, source_padding=True,
                             target_padding=True)
    train_data_loader = data_loader.data_loader()
    print("done")
