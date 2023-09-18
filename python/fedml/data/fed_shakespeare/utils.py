import numpy as np
import collections


word_dict = None
word_list = None
_pad = "<pad>"
_bos = "<bos>"
_eos = "<eos>"
"""
This code follows the steps of preprocessing in tff shakespeare dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/shakespeare_dataset.py
"""

SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017
# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
CHAR_VOCAB = list(
    "dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#'/37;?bfjnrvzBFJNRVZ\"&*.26:\naeimquyAEIMQUY]!%)-159\r"
)


def get_word_dict():
    """
    Get a dictionary mapping words to their corresponding IDs.

    Returns:
        collections.OrderedDict: A dictionary with words as keys and their IDs as values.
    """
    global word_dict
    if word_dict is None:
        words = [_pad] + CHAR_VOCAB + [_bos] + [_eos]
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict


def get_word_list():
    """
    Get a list of words in the vocabulary.

    Returns:
        list: A list of words in the vocabulary.
    """
    global word_list
    if word_list is None:
        word_dict = get_word_dict()
        word_list = list(word_dict.keys())
    return word_list


def id_to_word(idx):
    """
    Convert a word ID to the corresponding word.

    Args:
        idx (int): The word ID.

    Returns:
        str: The corresponding word.
    """
    return get_word_list()[idx]


def char_to_id(char):
    """
    Convert a character to its corresponding ID using the word_dict.

    Args:
        char (str): The character to convert.

    Returns:
        int: The corresponding ID for the character.
    """
    word_dict = get_word_dict()
    if char in word_dict:
        return word_dict[char]
    else:
        return len(word_dict)


def preprocess(sentences, max_seq_len=SEQUENCE_LENGTH):
    """
    Preprocess a list of sentences by converting characters to IDs and padding.

    Args:
        sentences (list): A list of sentences, where each sentence is a string.
        max_seq_len (int): Maximum sequence length (including start and end tokens).

    Returns:
        list: A list of sequences, where each sequence is a list of token IDs.
    """
    sequences = []

    def to_ids(sentence, num_oov_buckets=1):
        """
        Map a sentence to a list of token IDs and pad it to the specified length.

        Args:
            sentence (str): The input sentence.
            num_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets.
            max_seq_len (int): Maximum sequence length (including start and end tokens).

        Returns:
            list: A list of token IDs, padded to max_seq_len.
        """
        tokens = [char_to_id(c) for c in sentence]
        tokens = [char_to_id(_bos)] + tokens + [char_to_id(_eos)]
        if len(tokens) % (max_seq_len + 1) != 0:
            pad_length = (-len(tokens)) % (max_seq_len + 1)
            tokens += [char_to_id(_pad)] * pad_length
        return (
            tokens[i : i + max_seq_len + 1]
            for i in range(0, len(tokens), max_seq_len + 1)
        )

    for sen in sentences:
        sequences.extend(to_ids(sen))
    return sequences


def split(dataset):
    """
    Split a dataset into input sequences (x) and target sequences (y).

    Args:
        dataset (list): A list of sequences, where each sequence is a list of token IDs.

    Returns:
        tuple: A tuple containing two arrays, x and y, where x represents input sequences
        and y represents target sequences.
    """
    ds = np.asarray(dataset)
    x = ds[:, :-1]
    y = ds[:, 1:]
    return x, y



if __name__ == "__main__":
    print(
        split(
            preprocess(
                [
                    "Yonder comes my master, your brother.",
                    "Come not within these doors; within this roof\nThe enemy of all your graces lives.\nYour brother- no, no brother; yet the son-\nYet not the son; I will not call him son\nOf him I was about to call his father-\nHath heard your praises; and this night he means\nTo burn the lodging where you use to lie,\nAnd you within it. If he fail of that,\nHe will have other means to cut you off;\nI overheard him and his practices.\nThis is no place; this house is but a butchery;\nAbhor it, fear it, do not enter it.\nNo matter whither, so you come not here.",
                    "To the last gasp, with truth and loyalty.\nFrom seventeen years till now almost four-score\nHere lived I, but now live here no more.\nAt seventeen years many their fortunes seek,\nBut at fourscore it is too late a week;\nYet fortune cannot recompense me better\nThan to die well and not my master's debtor.          Exeunt\nDear master, I can go no further. O, I die for food! Here lie",
                    "[Coming forward] Sweet masters, be patient; for your father's",
                    "remembrance, be at accord.\nIs 'old dog' my reward? Most true, I have lost my teeth in",
                ]
            )
        )
    )
