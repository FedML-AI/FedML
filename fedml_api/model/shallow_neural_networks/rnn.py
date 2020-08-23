import torch.nn as nn


class RNN_OriginalFedAvg(nn.Module):
    """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
      This replicates the model structure in the paper:
      Communication-Efficient Learning of Deep Networks from Decentralized Data
        H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas. AISTATS 2017.
        https://arxiv.org/abs/1602.05629
      This is also recommended model by "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
      Args:
        vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
        sequence_length: the length of input sequences.
      Returns:
        An uncompiled `torch.nn.Module`.
      """

    def __init__(self, embedding_dim=8, vocab_size=90):
        super(RNN_OriginalFedAvg, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=256, num_layers=2)
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, input_seq, hidden_state):
        embeds = self.embeddings(input_seq)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        output = self.fc(lstm_out)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


class RNN_StackOverFlow(nn.Module):
    """Creates a RNN model using LSTM layers for StackOverFlow (next word prediction task).
      This replicates the model structure in the paper:
      "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
      Table 9
      Args:
        vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
        sequence_length: the length of input sequences.
      Returns:
        An uncompiled `torch.nn.Module`.
      """

    def __init__(self, vocab_size=10000,
                           num_oov_buckets=1,
                           embedding_size=96,
                           latent_size=670,
                           num_layers=1):
        super(RNN_StackOverFlow, self).__init__()
        extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
        self.word_embeddings = nn.Embedding(num_embeddings=extended_vocab_size, embedding_dim=embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=latent_size, num_layers=num_layers)
        self.fc1 = nn.Linear(latent_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, extended_vocab_size)

    def forward(self, input_seq, hidden_state):
        embeds = self.word_embeddings(input_seq)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        fc1_output = self.fc1(lstm_out)
        output = self.fc2(fc1_output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())
