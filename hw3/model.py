# IMPLEMENT YOUR MODEL CLASS HERE
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """


    def __init__(self, embedding_dim, input_len, vocab_size, encoding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm_e = torch.nn.LSTM(input_size=input_len*embedding_dim, hidden_size=encoding_dim, num_layers=1)

    def forward(self, x):
        embeds = self.embedding(x).view((x.shape[0], -1))
        h_d, (_, _) = self.lstm_e(embeds)
        return h_d



class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, encoding_dim, n_acts, n_targets):
        super().__init__()
        self.fca = torch.nn.Linear(encoding_dim, n_acts)
        self.fct = torch.nn.Linear(encoding_dim, n_targets)
        # self.lstm_d = torch.nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1)

    def forward(self, h_d):
        a_idx = self.fca(h_d)
        t_idx = self.fct(h_d)
        # torch.nn.functional.softmax(a_idx)
        # torch.nn.functional.softmax(t_idx)
        # self.decoder.lstm_d([a_idx, t_idx])
        return a_idx, t_idx


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, embedding_dim, n_acts, n_targets, n_voc, len_cutoff,
                 encoding_dim = 20):
        super().__init__()
        self.encoder = Encoder(embedding_dim, len_cutoff, n_voc, encoding_dim)
        self.decoder = Decoder(encoding_dim, n_acts, n_targets)
        # self.decoder.lstm_d = self.encoder.lstm_e

    def forward(self, x):
        h_d = self.encoder(x)
        a_idx, t_idx = self.decoder(h_d)
        return torch.hstack((a_idx, t_idx))

