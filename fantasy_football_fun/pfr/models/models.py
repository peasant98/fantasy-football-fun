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


class FantasyFootballLSTM(nn.Module):
    """
    LSTM model for fantasy football data.
    """
    def __init__(self, input_dim, hidden_dim, fc1_dim=64, categories=5):
        super(FantasyFootballLSTM, self).__init__()
        # 2 stacked lstm
        # many folks use softmax to predict token, not whole vector
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2)
        self.fc1 = nn.Linear(hidden_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, 128)
        self.embedding = nn.Embedding(categories, 3)
        self.position_fc = nn.Linear(15, 1)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, hidden, pos):
        x1 = self.lstm(x, hidden)
        x1 = self.fc1(x1[0])
        x1 = self.fc2(x1)
        x2 = self.embedding(pos.type(torch.LongTensor).cuda())
        x2 = torch.flatten(x2)
        x2 = self.position_fc(x2)

        x3 = torch.add(x1, x2)
        return self.fc3(x3)


# class FantasyFootballModel():
#     def __init__(self):
#         super().__init__()

# class MidiLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(MidiLSTM, self).__init__()
#         # 2 stacked lstm
#         # many folks use softmax to predict token, not whole vector
#         # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

#         self.lstm = nn.LSTM(input_dim, hidden_dim, 2)
#         self.fc = nn.Linear(hidden_dim, 128)

#     def forward(self, x, hidden):
#         x = self.lstm(x, hidden)
#         x = self.fc(x[0])

#         return x

# # embed to EMBEDDING_DIM dimension vectors
# # adding reduction="sum" will be make gradient updates more extreme
# loss_function = nn.BCEWithLogitsLoss()
# model = MidiLSTM(128, 256)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# md = MidiDataset("bach")

# # input is (num of sequences, batch size, time step)
# model.cuda()
# for epoch in range(50):
#     print("Epoch", epoch)
#     vals = list(range(len(md)))
#     np.random.shuffle(vals) 
#     for i, val in enumerate(vals):
#         hidden = (torch.randn(2, 1, 256).cuda(), torch.randn(2, 1, 256).cuda())
#         # values in dataset
#         optimizer.zero_grad()
#         pianoroll = md[i][:-1]
#         target = md[i][1:]


#         x = pianoroll.view(-1, 1, 128).cuda()
#         target = target.view(-1, 1, 128).cuda()
#         output = model(x, hidden)
#         loss = loss_function(output, target)
#         loss.backward(retain_graph=True)
#         if i % 10 == 0:
#             print("Loss:", loss.item())
#         optimizer.step()

if __name__ == '__main__':
    dataset = FFDataset.FantasyFootballDataset("../finalized_players.csv", normalize=True)
    print(dataset[1])
    # exit()

    career, pos_vec = dataset[1]

    model = FantasyFootballLSTM(1, 128)
    model.cuda()
    hidden = (torch.randn(2, 1, 128).cuda(), torch.randn(2, 1, 128).cuda())
    mycareer = career.view(-1, 1, 1)
    out = model(mycareer, hidden, pos_vec)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(50):
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"thirdmodel_{epoch}")
        losses = []
        print(f"NEW EPOCH {epoch}")
        vals = list(range(len(dataset)))
        np.random.shuffle(vals)
        for i, idx in enumerate(vals):
            hidden = (torch.randn(2, 1, 128).cuda(),
                      torch.randn(2, 1, 128).cuda())
            optimizer.zero_grad()
            inputs = dataset[idx][0][:-1]
            target = dataset[idx][0][1:]
            if len(inputs) == 0:
                continue
            inputs = inputs.view(-1, 1, 1)
            target = target.view(-1, 1, 1)
            pos_onehot = dataset[idx][1]
            out = model(inputs, hidden, pos_onehot)
            loss = loss_function(out, target)
            losses.append(loss.item())
            loss.backward()
            if i % 100 == 0:
                print("Loss:", loss.item())
                print(out)
            optimizer.step()
        mean_loss = np.mean(losses)
        print(f"Mean LOSS: {mean_loss}")
    #     # amount sentences

