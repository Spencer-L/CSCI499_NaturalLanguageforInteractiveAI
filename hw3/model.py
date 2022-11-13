# IMPLEMENT YOUR MODEL CLASS HERE
import torch
import torch.nn as nn

from transformers import BertConfig, BertModel

class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """


    def __init__(self, embedding_dim, input_len, vocab_size):
        super().__init__()
        # # embedding layer
        # self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        #
        # # maxpool layer
        # self.maxpool = torch.nn.MaxPool2d((input_len, 1), ceil_mode=True)

        self.lstm_e = torch.nn.LSTM(input_size=input_len, hidden_size=embedding_dim, num_layers=1)

    def forward(self, x):
        h_d, (_, _) = self.lstm_e(x)
        return h_d



class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, embedding_dim, n_acts, n_targets):
        super().__init__()
        self.fca = torch.nn.Linear(embedding_dim, n_acts)
        self.fct = torch.nn.Linear(embedding_dim, n_targets)
        # self.lstm_d = torch.nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1)

    def forward(self, h_d):
        a_idx = self.fca(h_d)
        t_idx = self.fct(h_d)
        # self.decoder.lstm_d([a_idx, t_idx])
        return a_idx, t_idx


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, embedding_dim, n_acts, n_targets, n_voc, len_cutoff):
        super().__init__()
        self.encoder = Encoder(embedding_dim, len_cutoff, n_voc)
        self.decoder = Decoder(embedding_dim, n_acts, n_targets)
        # self.decoder.lstm_d = self.encoder.lstm_e

    def forward(self, x):
        h_d = self.encoder(x)
        a_idx, t_idx = self.decoder(h_d)
        # breakpoint()
        return torch.hstack((a_idx, t_idx))

