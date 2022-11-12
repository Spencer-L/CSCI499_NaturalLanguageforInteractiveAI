# IMPLEMENT YOUR MODEL CLASS HERE
import torch.nn as nn

from transformers import BertConfig, BertModel

class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """


    def __init__(self):
        # # Building the config
        # config = BertConfig()
        # # Building the model from the config
        # self.bert = BertModel(config)


    def forward(self, x):
        # return self.bert(x)



class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self):
        pass

    def forward(self, x):
        pass


class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self):
        pass

    def forward(self, x):
        pass
