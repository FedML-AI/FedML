from nltk.tokenize import word_tokenize
import os
train_file_path = "../../../../data/fednlp/Cornell Movie--Dialogs Corpus/cornell movie-dialogs corpus/"
movie_conversation_file_name = "movie_conversations.txt"
movie_line_file_name = "movie_lines.txt"
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


def load_data(conversation_file_path, line_file_path, source_max_sequence_length=None, target_max_sequence_length=None,
              history_max_sequence_length=None, source_padding=True, target_padding=True, history_padding=True):
    line_dict = {}
    with open(line_file_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                temp = line.split("+++$+++")
                line_dict[temp[0].strip()] = {"utterance": temp[-1].strip(), "character_id": temp[1]}
    x = []
    y = []
    history = []
    source_sequence_lengths = []
    target_sequence_lengths = []
    history_sequence_lengths = []
    vocab = dict()
    conversation = []
    attributes = dict()
    attributes["inputs"] = []
    with open(conversation_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                temp = line.split("+++$+++")
                conversation_idx = temp[-1].strip()
                conversation_idx = eval(conversation_idx)
                for i in range(len(conversation_idx) - 1):
                    tokens = word_tokenize(line_dict[conversation_idx[i]]["utterance"])
                    character_id = line_dict[conversation_idx[i]]["character_id"]
                    next_tokens = word_tokenize(line_dict[conversation_idx[i+1]]["utterance"])
                    next_character_id = line_dict[conversation_idx[i]]["character_id"]
                    for token in tokens:
                        if token not in vocab:
                            vocab[token] = len(vocab)
                    for next_token in next_tokens:
                        if next_token not in vocab:
                            vocab[next_token] = len(vocab)
                    x.append(tokens.copy())
                    y.append(next_tokens.copy())
                    history.append(conversation.copy())
                    source_sequence_lengths.append(len(tokens))
                    target_sequence_lengths.append(len(next_tokens))
                    history_sequence_lengths.append(len(conversation))
                    conversation += tokens
                    attributes["inputs"].append({"character_id": character_id, "next_character_id": next_character_id,
                                                 "movie_id": temp[2]})
                conversation.clear()

    if source_max_sequence_length is None:
        source_max_sequence_length = max(source_sequence_lengths)

    if target_max_sequence_length is None:
        target_max_sequence_length = max(target_sequence_lengths)

    if history_max_sequence_length is None:
        history_max_sequence_length = max(history_sequence_lengths)

    if source_padding:
        vocab[pad_token] = len(vocab)
        padding_data(x, source_max_sequence_length)

    if target_padding:
        if pad_token not in vocab: vocab[pad_token] = len(vocab)
        padding_data(y, target_max_sequence_length)

    if history_padding:
        if pad_token not in vocab: vocab[pad_token] = len(vocab)
        padding_data(history, history_max_sequence_length)

    return x, y, history, source_max_sequence_length, target_max_sequence_length, history_max_sequence_length, \
           source_sequence_lengths, target_sequence_lengths, history_sequence_lengths, vocab, attributes


if __name__ == "__main__":
    x, y, history, source_max_sequence_length, target_max_sequence_length, history_max_sequence_length, \
    source_sequence_lengths, target_sequence_lengths, history_sequence_lengths, vocab, attributes = \
        load_data(os.path.join(train_file_path, movie_conversation_file_name),
                  os.path.join(train_file_path, movie_line_file_name))
    print("done")
