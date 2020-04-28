import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pretty_midi
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pypianoroll import Multitrack, Track
import pandas as pd

from fantasy_football_fun.pfr import FFDataset

# use embeddings
# using dense layer is crucial here to predict
# lstm layer is a feature layer, need dense layer for probability
# distribution


class ModifiedFantasyFootballLSTM(nn.Module):
    """
    LSTM model for fantasy football data. Data from positions
    is used before being fed into the LSTM.
    """
    def __init__(self, input_dim, hidden_dim, fc1_dim=64, categories=5):
        super(ModifiedFantasyFootballLSTM, self).__init__()
        # 2 stacked lstm
        # many folks use softmax to predict token, not whole vector
        self.embedding = nn.Embedding(categories, 2)
        self.position_fc = nn.Linear(2, 1)
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden, pos):
        # x2 = self.embedding(pos.type(torch.LongTensor).cuda())
        # x2 = F.relu(self.position_fc(x2))

        # x2 = torch.flatten(x2)
        # x = torch.add(x, x2)
        x1 = self.lstm(x, hidden)
        x1 = F.relu(self.fc1(x1[0]))
        # print(x1)
        return (x1)


class FantasyFootballLSTM(nn.Module):
    """
    LSTM model for fantasy football data.
    """
    def __init__(self, input_dim, hidden_dim, fc1_dim=64, categories=5):
        super(FantasyFootballLSTM, self).__init__()
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


# # embed to EMBEDDING_DIM dimension vectors
# # adding reduction="sum" will be make gradient updates more extreme

if __name__ == '__main__':
    dataset = FFDataset.FantasyFootballDataset("../finalized_players.csv", normalize=True)

    data_len = len(dataset)
    train_length = int(data_len * 0.9)
    test_length = data_len - train_length
    train_set, test_set = torch.utils.data.random_split(dataset, [train_length, test_length])
    model = ModifiedFantasyFootballLSTM(1, 128)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    total_epoch_train_losses = []
    total_epoch_test_losses = []
    for epoch in range(50):
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"eighthmodel_{epoch}")
        losses = []
        print(f"NEW EPOCH {epoch}")
        vals = list(range(len(train_set)))
        np.random.shuffle(vals)
        for i, idx in enumerate(vals):
            hidden = (torch.randn(1, 1, 128).cuda(),
                      torch.randn(1, 1, 128).cuda())
            optimizer.zero_grad()
            inputs = train_set[idx][0][:-1]
            target = train_set[idx][0][1:]
            if len(inputs) == 0:
                continue
            # mean = torch.mean(inputs)
            # if mean <= 0.2:
            #     continue
            inputs = inputs.view(-1, 1, 1)
            target = target.view(-1, 1, 1)
            pos_onehot = train_set[idx][1]
            out = model(inputs, hidden, pos_onehot)
            loss = loss_function(out, target)
            losses.append(loss.item())
            loss.backward()
            if i % 100 == 0:
                print("Loss:", loss.item())
                # print(out)
            optimizer.step()
        mean_loss = np.mean(losses)

        total_epoch_train_losses.append(mean_loss)
        print(f"Mean TRAIN LOSS: {mean_loss}")

        # get testing losses
        test_losses = []
        for idx in range(len(test_set)):
            hidden = (torch.randn(1, 1, 128).cuda(),
                      torch.randn(1, 1, 128).cuda())
            inputs = test_set[idx][0][:-1]
            target = test_set[idx][0][1:]
            if len(inputs) == 0:
                continue
            # mean = np.mean(inputs)
            # if mean <= 0.2:
            #     continue
            inputs = inputs.view(-1, 1, 1)
            target = target.view(-1, 1, 1)
            pos_onehot = train_set[idx][1]
            out = model(inputs, hidden, pos_onehot)
            loss = loss_function(out, target)
            test_losses.append(loss.item())
        test_mean = np.mean(test_losses)
        total_epoch_test_losses.append(test_mean)
        print(f'TEST LOSS: {test_mean}')
    np.savetxt('train_loss2.txt', np.array(total_epoch_train_losses))
    np.savetxt('test_loss2.txt', np.array(total_epoch_test_losses))

