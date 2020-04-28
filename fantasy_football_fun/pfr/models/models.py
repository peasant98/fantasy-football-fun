import torch
import torch.nn as nn
import torch.nn.functional as F


# lstm layer is a feature layer, need dense layer for probability distribution

class FantasyFootballLSTM(nn.Module):
    """
    LSTM model for fantasy football data. Data from positions
    is used before being fed into the LSTM.
    Doesn't use position in the data
    """
    def __init__(self, input_dim, hidden_dim, fc1_dim=64):
        super(FantasyFootballLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1)
        # only need to predict one value
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden, pos):
        # position not used here
        x1 = self.lstm(x, hidden)
        x1 = (self.fc1(x1[0]))
        return x1


class AdvancedFantasyFootballLSTM(nn.Module):
    """
    LSTM model for fantasy football data. Uses position as a factor
    into the data.
    """
    def __init__(self, input_dim, hidden_dim, fc1_dim=64, categories=5):
        super(AdvancedFantasyFootballLSTM, self).__init__()
        # 2 stacked lstm
        # many folks use softmax to predict token, not whole vector
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, 128)
        self.embedding = nn.Embedding(categories, 3)
        self.position_fc = nn.Linear(15, 1)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, hidden, pos):
        x1 = self.lstm(x, hidden)
        x1 = F.relu(self.fc1(x1[0]))
        x1 = F.relu(self.fc2(x1))
        x2 = self.embedding(pos.type(torch.LongTensor).cuda())
        x2 = torch.flatten(x2)
        x2 = F.relu(self.position_fc(x2))
        x3 = torch.add(x1, x2)
        return nn.Sigmoid()(self.fc3(x3))
