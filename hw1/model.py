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
         n_acts,
         n_locs,
         embedding_dim
    ):
        super(Model, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.n_acts = n_acts
        self.n_locs = n_locs

        # embedding layer
        # self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        # # 

        # # linear layer
        # self.linear = torch.nn.Linear(2, n_acts+n_locs)

        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # maxpool layer
        self.maxpool = torch.nn.MaxPool2d((input_len, 1), ceil_mode=True)


        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=embedding_dim,
                                  hidden_size=2,
                                  num_layers=1)
        
        # linear layer
        self.fc = torch.nn.Linear(2, n_locs+n_acts)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # batch_size, seq_len = x.size(0), x.size(1)

        # h1 = self.embedding(x)
        # # maxpooled_embeds = self.maxpool(embeds)
        # h2, (_, _) = self.lstm(h1)
        # h3 = self.linear(h2)
        # out = self.sigmoid(h3)
        # breakpoint()

        batch_size, seq_len = x.size(0), x.size(1)

        embeds = self.embedding(x)
        maxpooled_embeds = self.maxpool(embeds)
        lstm_out, (_, _) = self.lstm(maxpooled_embeds)
        # squeeze out the singleton length dimension that we maxpool'd over
        linear_out = self.fc(lstm_out.squeeze(1))
        # out = lstm_out.squeeze(1)
        out = self.sigmoid(linear_out)
        # breakpoint()
        return out[:,:self.n_acts], out[:,self.n_acts:]