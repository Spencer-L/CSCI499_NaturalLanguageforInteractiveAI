import torch.nn as nn
EMBED_DIMENSION = 50
EMBED_MAX_NORM = 1
class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = EMBED_DIMENSION
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs_):
        x = self.embed(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        # x = self.softmax(x)
        return x