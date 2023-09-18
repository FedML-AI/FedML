# Variables
from spacy.lang.cs import Czech
from spacy.lang.ru import Russian
from spacy.lang.en import English
from spacy.lang.zh import Chinese
from spacy.lang.de import German
import spacy

from .globals import *

import gensim
import h5py
import json
import numpy as np


FLOAT_SIZE = 4


class SpacyTokenizer:
    """Tokenizer class for different languages using spaCy models.

    Attributes:
        __zh_tokenizer: Chinese tokenizer instance.
        __en_tokenizer: English tokenizer instance.
        __cs_tokenizer: Czech tokenizer instance.
        __de_tokenizer: German tokenizer instance.
        __ru_tokenizer: Russian tokenizer instance.

    Methods:
        get_tokenizer(lang): Get a spaCy tokenizer for the specified language.
    """
    def __init__(self):
        self.__zh_tokenizer = None
        self.__en_tokenizer = None
        self.__cs_tokenizer = None
        self.__de_tokenizer = None
        self.__ru_tokenizer = None

    @staticmethod
    def get_tokenizer(lang):
        """Get a spaCy tokenizer for the specified language.

        Args:
            lang (str): The language code (e.g., "zh" for Chinese, "en" for English).

        Returns:
            spacy.language.Language: A spaCy tokenizer instance.
        
        Raises:
            Exception: If an unacceptable language code is provided.
        """
        if lang == "zh":
            # nlp = spacy.load("zh_core_web_sm")
            nlp = Chinese()
        elif lang == "en":
            # nlp = spacy.load("en_core_web_sm")
            nlp = English()
        elif lang == "cs":
            nlp = Czech()
        elif lang == "de":
            # nlp = spacy.load("de_core_web_sm")
            nlp = German()
        elif lang == "ru":
            nlp = Russian()
        else:
            raise Exception("Unacceptable language.")
        return nlp

    @property
    def zh_tokenizer(self):
        """Chinese tokenizer property."""
        if self.__zh_tokenizer is None:
            self.__zh_tokenizer = self.get_tokenizer("zh")
        return self.__zh_tokenizer

    @property
    def en_tokenizer(self):
        """English tokenizer property."""
        if self.__en_tokenizer is None:
            self.__en_tokenizer = self.get_tokenizer("en")
        return self.__en_tokenizer

    @property
    def cs_tokenizer(self):
        """Czech tokenizer property."""
        if self.__cs_tokenizer is None:
            self.__cs_tokenizer = self.get_tokenizer("cs")
        return self.__cs_tokenizer

    @property
    def de_tokenizer(self):
        """German tokenizer property."""
        if self.__de_tokenizer is None:
            self.__de_tokenizer = self.get_tokenizer("de")
        return self.__de_tokenizer

    @property
    def ru_tokenizer(self):
        """Russian tokenizer property."""
        if self.__ru_tokenizer is None:
            self.__ru_tokenizer = self.get_tokenizer("ru")
        return self.__ru_tokenizer


def build_vocab(x):
    """Build a vocabulary from a list of tokenized sequences.

    Args:
        x (list): List of tokenized sequences, where each sequence is a list of tokens.

    Returns:
        dict: A vocabulary where tokens are keys and their corresponding indices are values.
    """
    vocab = dict()
    for single_x in x:
        for token in single_x:
            if token not in vocab:
                vocab[token] = len(vocab)
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    return vocab


def build_freq_vocab(x):
    """Build a frequency-based vocabulary from a list of tokenized sequences.

    Args:
        x (list): List of tokenized sequences, where each sequence is a list of tokens.

    Returns:
        dict: A vocabulary where tokens are keys and their frequencies are values.
    """
    freq_vocab = dict()
    for single_x in x:
        for token in single_x:
            if token not in freq_vocab:
                freq_vocab[token] = 1
            else:
                freq_vocab[token] += 1
    return freq_vocab


def padding_data(x, max_sequence_length):
    """Pad sequences in a list to a specified maximum sequence length.

    Args:
        x (list): List of sequences, where each sequence is a list of tokens.
        max_sequence_length (int): The desired maximum sequence length for padding.

    Returns:
        list: Padded sequences with a length of max_sequence_length.
        list: Sequence lengths before padding.
    """
    padding_x = []
    seq_lens = []
    for single_x in x:
        new_single_x = single_x.copy()
        if len(new_single_x) <= max_sequence_length:
            seq_lens.append(len(new_single_x))
            for _ in range(len(new_single_x), max_sequence_length):
                new_single_x.append(PAD_TOKEN)
        else:
            seq_lens.append(max_sequence_length)
            new_single_x = new_single_x[:max_sequence_length]
        padding_x.append(new_single_x)
    return padding_x, seq_lens


def padding_char_data(x, max_sequence_length, max_word_length):
    """Pad character-level sequences in a list to specified maximum lengths.

    Args:
        x (list): List of sequences, where each sequence is a list of character tokens.
        max_sequence_length (int): The desired maximum sequence length for padding.
        max_word_length (int): The desired maximum word length for character tokens.

    Returns:
        list: Padded character sequences with specified word and sequence lengths.
        list: Word lengths before padding.
    """
    padding_x = []
    word_lens = []
    for sent in x:
        new_sent = []
        temp_word_lens = []
        for chars in sent:
            new_chars = chars.copy()
            if len(new_chars) <= max_word_length:
                temp_word_lens.append(len(new_chars))
                for _ in range(len(new_chars), max_word_length):
                    new_chars.append(PAD_TOKEN)
            else:
                temp_word_lens.append(max_word_length)
                new_chars = new_chars[:max_word_length]
            new_sent.append(new_chars)
        if len(new_sent) <= max_sequence_length:
            for _ in range(len(new_sent), max_sequence_length):
                new_sent.append([PAD_TOKEN for _ in range(max_word_length)])
        else:
            new_sent = new_sent[:max_sequence_length]
            temp_word_lens = temp_word_lens[:max_sequence_length]
        word_lens.append(temp_word_lens)
        padding_x.append(new_sent)
    return padding_x, word_lens


def token_to_idx(x, vocab):
    """Convert tokenized sequences to indices using a vocabulary.

    Args:
        x (list): List of tokenized sequences, where each sequence is a list of tokens.
        vocab (dict): A vocabulary where tokens are keys and their corresponding indices are values.

    Returns:
        list: Sequences with tokens replaced by their corresponding indices.
    """
    idx_x = []
    for single_x in x:
        new_single_x = []
        for token in single_x:
            if token in vocab:
                new_single_x.append(vocab[token])
            else:
                new_single_x.append(vocab[UNK_TOKEN])
        idx_x.append(new_single_x)
    return idx_x


def char_to_idx(x, vocab):
    idx_x = []
    for sent in x:
        new_sent = []
        for token in sent:
            new_token = []
            for ch in token:
                if ch in vocab:
                    new_token.append(vocab[ch])
                else:
                    new_token.append(vocab[UNK_TOKEN])
            new_sent.append(new_token)
        idx_x.append(new_sent)
    return idx_x


def label_to_idx(y, vocab):
    idx_y = []
    for label in y:
        idx_y.append(vocab[label])
    return idx_y


def remove_words(x, removed_words):
    remove_x = []
    for single_x in x:
        new_single_x = []
        for token in single_x:
            if token not in removed_words:
                new_single_x.append(token)
        remove_x.append(new_single_x)
    return remove_x


def load_word2vec_embedding(path, source_vocab):
    vocab = dict()
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    weights = []
    for key, value in model.vocab.items():
        if source_vocab is not None and key in source_vocab:
            vocab[key] = len(vocab)
            weights.append(model.vectors[value.index])
        else:
            vocab[key] = len(vocab)
            weights.append(model.vectors[value.index])
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    weights.append(np.zeros(model.vector_size))
    weights.append(np.zeros(model.vector_size))
    weights = np.array(weights)
    return vocab, weights


def load_glove_embedding(path, source_vocab, dimension):
    vocab = dict()
    weights = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            temp = line.split(" ")
            word = " ".join(temp[:-dimension])
            if source_vocab is not None:
                if word in source_vocab:
                    vocab[word] = len(vocab)
                    weights.append(np.array([float(num) for num in temp[-dimension:]]))
            else:
                vocab[word] = len(vocab)
                weights.append(np.array([float(num) for num in temp[-dimension:]]))
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    weights.append(np.zeros(dimension))
    weights.append(np.zeros(dimension))
    weights = np.array(weights)
    return vocab, weights


def NER_data_formatter(ner_data):
    formatted_data = []
    if len(ner_data["X"]) != len(ner_data["Y"]):
        print(ner_data["X"])
        print(ner_data["Y"])
        print(len(ner_data["X"]), len(ner_data["Y"]))
    assert len(ner_data["X"]) == len(ner_data["Y"])
    sent_id = 0
    for x, y in zip(ner_data["X"], ner_data["Y"]):
        assert len(x) == len(y)
        for token, tag in zip(x, y):
            formatted_data.append([sent_id, token, tag])
        sent_id += 1
    return formatted_data


def generate_h5_from_dict(file_name, data_dict):
    """Generate an HDF5 file from a nested dictionary.

    Args:
        file_name (str): The name of the HDF5 file to be created.
        data_dict (dict): The nested dictionary containing data to be stored in the HDF5 file.

    Returns:
        None
    """
    def dict_to_h5_recursive(h5_file, path, dic):
        for key, value in dic.items():
            if isinstance(value, dict):
                if key == "attributes":
                    h5_file[path + str(key)] = json.dumps(value)
                else:
                    dict_to_h5_recursive(h5_file, path + str(key) + "/", value)
            else:
                if isinstance(value, list) and (
                    len(value) > 0 and isinstance(value[0], str)
                ):
                    h5_file[path + str(key)] = np.array(
                        [v.encode("utf8") for v in value], dtype="S"
                    )
                else:
                    h5_file[path + str(key)] = value

    f = h5py.File(file_name, "w")
    dict_to_h5_recursive(f, "/", data_dict)
    f.close()


def decode_data_from_h5(data):
    """Decode data from bytes to UTF-8 string if necessary.

    Args:
        data (bytes or any): The input data, which may be in bytes.

    Returns:
        str or any: The decoded data as a UTF-8 string, or the input data if it's not in bytes.
    """
    if isinstance(data, bytes):
        return data.decode("utf8")
    return data
