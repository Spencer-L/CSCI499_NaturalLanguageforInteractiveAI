import torch
import torchinfo

batch_size = 32
timesteps = 12
input_features = 16
h1_features = 8
h2_features = 4
h3_features = 2
output_features = 1

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(input_size=input_features, hidden_size=h1_features)
        self.lstm2 = torch.nn.LSTM(input_size=h1_features, hidden_size=h2_features)
        self.fc1 = torch.nn.Linear(h2_features, h3_features)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(h3_features, output_features)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        h1, (h1_T,c1_T) = self.lstm1(inputs)
        h2, (h2_T, c2_T) = self.lstm2(h1)
        h3 = self.fc1(h2[-1,:,:])
        h3 = self.relu(h3)
        output = self.fc2(h3)
        output = self.sigmoid(output)
        return output

model = SimpleModel()

torchinfo.summary(model,(timesteps, batch_size, input_features))