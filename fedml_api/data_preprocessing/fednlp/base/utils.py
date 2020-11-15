

def build_vocab(x):
    vocab = dict()
    for single_x in x:
        for token in single_x:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
