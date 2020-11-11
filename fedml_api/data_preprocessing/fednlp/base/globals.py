# Constants

# Data Loader
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_LABEL = "O"
UNK_LABEL = "O"

# Partition
N_CLIENTS = 100
PARTITION_KEYS = ("X", "Y", "history", "sequence_length", "source_sequence_length", "target_sequence_length",
                  "history_sequence_length")
# Variables
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.zh import Chinese
from spacy.lang.de import German
from spacy.lang.ru import Russian
from spacy.lang.cs import Czech


class SpacyTokenizer:
    def __init__(self):
        self.__zh_tokenizer = None
        self.__en_tokenizer = None
        self.__cs_tokenizer = None
        self.__de_tokenizer = None
        self.__ru_tokenizer = None

    @staticmethod
    def get_tokenizer(lang):
        if lang == "zh":
            nlp = Chinese()
        elif lang == "en":
            nlp = English()
        elif lang == "cs":
            nlp = Czech()
        elif lang == "de":
            nlp = German()
        elif lang == "ru":
            nlp = Russian()
        else:
            raise Exception("Unacceptable language.")
        tokenizer = Tokenizer(nlp.vocab)
        return tokenizer

    @property
    def zh_tokenizer(self):
        if self.__zh_tokenizer is None:
            self.__zh_tokenizer = self.get_tokenizer("zh")
        return self.__zh_tokenizer

    @property
    def en_tokenizer(self):
        if self.__en_tokenizer is None:
            self.__en_tokenizer = self.get_tokenizer("en")
        return self.__en_tokenizer

    @property
    def cs_tokenizer(self):
        if self.__cs_tokenizer is None:
            self.__cs_tokenizer = self.get_tokenizer("cs")
        return self.__cs_tokenizer

    @property
    def de_tokenizer(self):
        if self.__de_tokenizer is None:
            self.__de_tokenizer = self.get_tokenizer("de")
        return self.__de_tokenizer

    @property
    def ru_tokenizer(self):
        if self.__ru_tokenizer is None:
            self.__ru_tokenizer = self.get_tokenizer("ru")
        return self.__ru_tokenizer


spacy_tokenizer = SpacyTokenizer()
