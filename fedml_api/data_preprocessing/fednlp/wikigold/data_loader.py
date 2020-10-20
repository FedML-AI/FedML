
train_file_path = "../../../../data/fednlp/wikigold/wikigold/CONLL-format/data/wikigold.conll.txt"
pad_token = "<PAD>"
pad_label = "O"


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
            idx_single_x.append(token_vocab[token])
            idx_single_y.append(label_vocab[label])
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

    if padding:
        token_vocab[pad_token] = len(token_vocab)
        label_vocab[pad_label] = len(label_vocab)
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                token, label = line.split(" ")
                single_x.append(token)
                single_y.append(label)
                if token not in token_vocab:
                    token_vocab[token] = len(token_vocab)
                if label not in label_vocab:
                    label_vocab[label] = len(label_vocab)
            else:
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
    x, y, max_sequence_length, sequence_lengths, token_vocab, label_vocab = load_data(train_file_path)
    idx_x, idx_y = raw_data_to_idx(x, y, token_vocab, label_vocab)
    print("done")
