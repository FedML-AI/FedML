# use conll format to load the data
import os

train_file_path = "../../../../data/fednlp/W-NUT 2017/data/train_data/Conll_Format/"
dev_file_path = "../../../../data/fednlp/W-NUT 2017/data/dev_data/Conll_Format/"
test_file_path = "../../../../data/fednlp/W-NUT 2017/data/test_data/Conll_Format/"
test_2020_file_path = "../../../../data/fednlp/W-NUT 2017/data/test_data_2020/Conll_Format/"
pad_token = "<PAD>"
pad_label = "O"
unk_token = "<UNK>"
unk_label = "O"


def padding_data(x, y, max_sequence_length):
    assert len(x) == len(y)
    for i, single_x in enumerate(x):
        single_y = y[i]
        assert len(single_x) == len(single_y)
        if len(single_x) <= max_sequence_length:
            for _ in range(len(single_x), max_sequence_length):
                single_x.append(pad_token)
                single_y.append(pad_label)
        else:
            single_x = single_x[:max_sequence_length]
            single_y = single_y[:max_sequence_length]


def raw_data_to_idx(x, y, token_vocab, label_vocab):
    assert len(x) == len(y)
    idx_x = []
    idx_y = []
    for i, single_x in enumerate(x):
        single_y = y[i]
        assert len(single_x) == len(single_y)
        idx_single_x = []
        idx_single_y = []
        for j, token in enumerate(single_x):
            label = single_y[j]
            idx_single_x.append(token_vocab[token] if token in token_vocab else token_vocab[unk_token])
            idx_single_y.append(label_vocab[label] if label in label_vocab else label_vocab[unk_label])
        idx_x.append(idx_single_x)
        idx_y.append(idx_single_y)
    return idx_x, idx_y


def load_data(file_path, max_sequence_length=None, padding=True):
    x = []
    y = []
    sequence_lengths = []
    token_vocab = dict()
    label_vocab = dict()
    single_x = []
    single_y = []

    token_vocab[unk_token] = len(token_vocab)
    if padding:
        token_vocab[pad_token] = len(token_vocab)
        label_vocab[pad_label] = len(label_vocab)

    for root, dirs, files in os.walk(file_path):
        for name in files:
            path = os.path.join(root, name)
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        token, label = line.split("\t")
                        single_x.append(token)
                        single_y.append(label)
                        if token not in token_vocab:
                            token_vocab[token] = len(token_vocab)
                        if label not in label_vocab:
                            label_vocab[label] = len(label_vocab)
                    else:
                        if len(single_x) != 0:
                            assert len(single_x) == len(single_y)
                            x.append(single_x.copy())
                            y.append(single_y.copy())
                            sequence_lengths.append(len(single_x))
                        single_x.clear()
                        single_y.clear()
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    if padding:
        padding_data(x, y, max_sequence_length)
    return x, y, max_sequence_length, sequence_lengths, token_vocab, label_vocab


if __name__ == "__main__":
    train_x, train_y, train_max_sequence_length, train_sequence_lengths, train_token_vocab, train_label_vocab \
        = load_data(train_file_path)
    dev_x, dev_y, dev_max_sequence_length, dev_sequence_lengths, dev_token_vocab, dev_label_vocab \
        = load_data(dev_file_path)
    test_x, test_y, test_max_sequence_length, test_sequence_lengths, test_token_vocab, test_label_vocab \
        = load_data(test_file_path)
    test_2020_x, test_2020_y, test_2020_max_sequence_length, test_2020_sequence_lengths, test_2020_token_vocab, \
    test_2020_label_vocab = load_data(test_file_path)

    train_idx_x, train_idx_y = raw_data_to_idx(train_x, train_y, train_token_vocab, train_label_vocab)
    dev_idx_x, dev_idx_y = raw_data_to_idx(dev_x, dev_y, train_token_vocab, train_label_vocab)
    test_idx_x, test_idx_y = raw_data_to_idx(test_x, test_y, train_token_vocab, train_label_vocab)
    test_2020_idx_x, test_2020_idx_y = raw_data_to_idx(test_2020_x, test_2020_y, train_token_vocab, train_label_vocab)
    print("done")
