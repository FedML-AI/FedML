import torch
import torch.nn as nn


class RNN_OriginalFedAvg(nn.Module):
    """
    Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
    This replicates the model structure in the paper:
    Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
    This is also recommended model by "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
    
    Args:
        embedding_dim: The dimension of word embeddings. Default is 8.
        vocab_size: The size of the vocabulary, used as a dimension in the input embedding. Default is 90.
        hidden_size: The size of the hidden state in the LSTM layers. Default is 256.
    Returns:
        An uncompiled `torch.nn.Module`.
    """

    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(RNN_OriginalFedAvg, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        """
        Forward pass of the model.

        Args:
            input_seq: Input sequence of character indices.
        Returns:
            output: Model predictions.
        """
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        # For fed_shakespeare
        # output = self.fc(lstm_out[:,:])
        # output = torch.transpose(output, 1, 2)
        return output


class RNN_FedShakespeare(nn.Module):
    """
    RNN model for Shakespeare language modeling (next character prediction task).

    This class defines an RNN model for predicting the next character in a sequence of text,
    specifically tailored for the "fed_shakespeare" task.

    Args:
        embedding_dim (int): Dimension of the character embeddings.
        vocab_size (int): Size of the vocabulary (number of unique characters).
        hidden_size (int): Size of the hidden state of the LSTM layers.

    Returns:
        torch.Tensor: The model's output predictions.
    """
    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(RNN_FedShakespeare, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        """
        Forward pass of the model.

        Args:
            input_seq: Input sequence of character indices.
        Returns:
            output: Model predictions.
        """
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        # output = self.fc(final_hidden_state)
        # For fed_shakespeare
        output = self.fc(lstm_out[:, :])
        output = torch.transpose(output, 1, 2)
        return output


class RNN_StackOverFlow(nn.Module):
    """
    RNN model for StackOverflow language modeling (next word prediction task).

    This class defines an RNN model for predicting the next word in a sequence of text, specifically tailored
    for the "stackoverflow_nwp" task.
    "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)

    Args:
        vocab_size (int): Size of the vocabulary (number of unique words).
        num_oov_buckets (int): Number of out-of-vocabulary (OOV) buckets.
        embedding_size (int): Dimension of the word embeddings.
        latent_size (int): Size of the LSTM hidden state.
        num_layers (int): Number of LSTM layers.

    Returns:
        torch.Tensor: The model's output predictions.
    """

    def __init__(
            self,
            vocab_size=10000,
            num_oov_buckets=1,
            embedding_size=96,
            latent_size=670,
            num_layers=1,
    ):
        super(RNN_StackOverFlow, self).__init__()
        extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
        self.word_embeddings = nn.Embedding(
            num_embeddings=extended_vocab_size,
            embedding_dim=embedding_size,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_size, hidden_size=latent_size, num_layers=num_layers
        )
        self.fc1 = nn.Linear(latent_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, extended_vocab_size)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the model.

        Args:
            input_seq (torch.Tensor): Input sequence of word indices.
            hidden_state (tuple): Initial hidden state of the LSTM.

        Returns:
            torch.Tensor: Model predictions.
        """
        embeds = self.word_embeddings(input_seq)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        fc1_output = self.fc1(lstm_out[:, :])
        output = self.fc2(fc1_output)
        output = torch.transpose(output, 1, 2)
        return output
