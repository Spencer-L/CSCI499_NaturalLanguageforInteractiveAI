import torch.nn as nn
emb_dim = 10
emb_max_norm = 1
class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = emb_dim
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            max_norm=emb_max_norm,
        )
        self.linear = nn.Linear(
            in_features=emb_dim,
            out_features=vocab_size,
        )
        self.softmax = nn.Softmax(dim=1)
