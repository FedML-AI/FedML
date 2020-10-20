from nltk.tokenize import word_tokenize
import jieba
import functools


train_file_path = "../../../../data/fednlp/WMT/training-parallel-nc-v13/news-commentary-v13.{}-{}.{}"
language_pairs = [("cs", "en"), ("de", "en"), ("ru", "en"), ("zh", "en")]

tokenizer_dict = {"cs": functools.partial(word_tokenize, language="czech"),
                 "de": functools.partial(word_tokenize, language="german"),
                 "ru": functools.partial(word_tokenize, language="russian"),
                 "en": functools.partial(word_tokenize, language="english"),
                 "zh": jieba.cut}
pad_token = "<PAD>"
unk_token = "<UNK>"
sos_token = "<SOS>"
eos_token = "<EOS>"


def padding_data(x, max_sequence_length):
    for i, single_x in enumerate(x):
        if len(single_x) <= max_sequence_length:
            for _ in range(len(single_x), max_sequence_length):
                single_x.append(pad_token)
        else:
            single_x = single_x[:max_sequence_length]


def raw_data_to_idx(x, token_vocab):
    idx_x = []
    for i, single_x in enumerate(x):
        idx_single_x = []
        for j, token in enumerate(single_x):
            idx_single_x.append(token_vocab[token] if token in token_vocab else token_vocab[unk_token])
        idx_x.append(idx_single_x)
    return idx_x


def load_data(file_path, language_pair, source_max_sequence_length=None, target_max_sequence_length=None,
              source_padding=True, target_padding=True, sos_and_eos=True):
    x = []
    y = []
    source_sequence_lengths = []
    target_sequence_lengths = []
    source_token_vocab = dict()
    target_token_vocab = dict()

    with open(file_path.format(language_pair[0], language_pair[1], language_pair[0]), "r") as f:
        for line in f:
            line = line.strip()
            tokenizer = tokenizer_dict[language_pair[0]]
            tokens = tokenizer(line)
            for token in tokens:
                if token not in source_token_vocab:
                    source_token_vocab[token] = len(source_token_vocab)
            x.append(tokens)
            source_sequence_lengths.append(len(tokens))

    with open(file_path.format(language_pair[0], language_pair[1], language_pair[1]), "r") as f:
        for line in f:
            line = line.strip()
            tokenizer = tokenizer_dict[language_pair[0]]
            tokens = tokenizer(line)
            for token in tokens:
                if token not in target_token_vocab:
                    target_token_vocab[token] = len(target_token_vocab)
            y.append(tokens)
            target_sequence_lengths.append(len(tokens))

    if source_max_sequence_length is None:
        source_max_sequence_length = max(source_sequence_lengths)

    if target_max_sequence_length is None:
        target_max_sequence_length = max(target_sequence_lengths)

    if sos_and_eos:
        source_token_vocab[sos_token] = len(source_token_vocab)
        source_token_vocab[eos_token] = len(source_token_vocab)

        target_token_vocab[sos_token] = len(target_token_vocab)
        target_token_vocab[eos_token] = len(target_token_vocab)

        source_max_sequence_length += 2
        target_max_sequence_length += 2

        def add_sos_and_eos_token(x):
            for single_x in x:
                single_x = [sos_token] + single_x + [eos_token]

        add_sos_and_eos_token(x)
        add_sos_and_eos_token(y)

    if source_padding:
        source_token_vocab[pad_token] = len(source_token_vocab)
        padding_data(x, source_max_sequence_length)

    if target_padding:
        target_token_vocab[pad_token] = len(target_token_vocab)
        padding_data(y, target_max_sequence_length)

    return x, y, source_max_sequence_length, target_max_sequence_length, source_sequence_lengths, \
           target_sequence_lengths, source_token_vocab, target_token_vocab


if __name__ == "__main__":
    x, y, source_max_sequence_length, target_max_sequence_length, source_sequence_lengths, \
    target_sequence_lengths, source_token_vocab, target_token_vocab = load_data(train_file_path, language_pairs[0])
    print("done")
