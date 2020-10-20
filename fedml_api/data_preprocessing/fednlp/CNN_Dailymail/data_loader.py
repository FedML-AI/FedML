import struct
from tensorflow.core.example import example_pb2
train_file_path = "../../../../data/fednlp/CNN_Dailymail/finished_files/train.bin"
dev_file_path = "../../../../data/fednlp/CNN_Dailymail/finished_files/val.bin"
test_file_path = "../../../../data/fednlp/CNN_Dailymail/finished_files/test.bin"
vocab_file_path = "../../../../data/fednlp/CNN_Dailymail/finished_files/vocab"
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


def load_data(file_path, source_max_sequence_length=None, target_max_sequence_length=None, source_padding=True,
              target_padding=True):
    x = []
    y = []
    source_sequence_lengths = []
    target_sequence_lengths = []
    single_x = []
    single_y = []

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
        for token in article_text.split(" "):
            token = token.strip()
            if token:
                single_x.append(token)

        for token in abstract_text.split(" "):
            token = token.strip()
            if token:
                single_y.append(token)

        if len(single_x) != 0 and len(single_y) != 0:
            x.append(single_x.copy())
            y.append(single_y.copy())
            source_sequence_lengths.append(len(single_x))
            target_sequence_lengths.append(len(single_y))

        single_x.clear()
        single_y.clear()

    if source_max_sequence_length is None:
        source_max_sequence_length = max(source_sequence_lengths)
    if target_max_sequence_length is None:
        target_max_sequence_length = max(target_sequence_lengths)

    if source_padding:
        padding_data(x, source_max_sequence_length)
    if target_padding:
        padding_data(y, target_max_sequence_length)

    return x, y, source_max_sequence_length, target_max_sequence_length


def load_vocab(file_path, padding=True, start_and_end_token=True):
    token_vocab = dict()
    token_vocab[unk_token] = len(token_vocab)
    if padding:
        token_vocab[pad_token] = len(token_vocab)
    if start_and_end_token:
        token_vocab[sos_token] = len(token_vocab)
        token_vocab[eos_token] = len(token_vocab)
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            token, index = line.split(" ")
            if token not in token_vocab:
                token_vocab[token] = len(token_vocab)
    return token_vocab



if __name__ == "__main__":
    train_x, train_y, train_source_max_sequence_length, train_target_max_sequence_length = load_data(train_file_path)
    dev_x, dev_y, dev_source_max_sequence_length, dev_target_max_sequence_length = load_data(dev_file_path)
    test_x, test_y, test_source_max_sequence_length, test_target_max_sequence_length = load_data(test_file_path)
    vocab = load_vocab(vocab_file_path)
    train_idx_x, train_idx_y, dev_idx_x, dev_idx_y, dev_test_x, dev_test_y = raw_data_to_idx(train_x, vocab), \
                                                                             raw_data_to_idx(train_y, vocab), \
                                                                             raw_data_to_idx(dev_x, vocab), \
                                                                             raw_data_to_idx(dev_y, vocab), \
                                                                             raw_data_to_idx(test_x, vocab), \
                                                                             raw_data_to_idx(test_y, vocab)
    print("done")
