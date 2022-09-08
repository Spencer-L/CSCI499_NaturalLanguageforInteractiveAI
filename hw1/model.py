import torch

batch_size = 32
timesteps = 12
input_features = 161
h1_features = 8
h2_features = 4
h3_features = 2
output_features = 1

class Model(torch.nn.Module):
    def __init__(self,
         device,
         vocab_size,
         input_len,
         n_books,
         embedding_dim
    ):
        super(Model, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.n_books = n_books

        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # maxpool layer
        self.maxpool = torch.nn.MaxPool2d((input_len, 1), ceil_mode=True)

        # linear layer
        self.fc = torch.nn.Linear(embedding_dim, n_books)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        embeds = self.embedding(x)
        maxpooled_embeds = self.maxpool(embeds)
        out = self.fc(maxpooled_embeds).squeeze(1)  # squeeze out the singleton length dimension that we maxpool'd over

        return out