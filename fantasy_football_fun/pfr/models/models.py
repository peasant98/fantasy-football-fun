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


class FantasyFootballDataset(Dataset):
    def __init__(self, root_folder, csv, transpose=False, is_conv=False):
        # get the list of directories
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # over 30k entries long
        self.data_frame = pd.read_csv(os.path.join(root_folder, csv))
        self.tracks_dict = {}
        self.is_conv = is_conv
        self.beat_resolution = 4
        dirs = [d for d in os.listdir(root_folder) 
                    if os.path.isdir(os.path.join(root_folder, d))]

        # get all files
        self.midi_files = []
        accum = []
        for d in dirs: 
            midi_names = os.listdir(os.path.join(root_folder, d))
            for i in range(len(midi_names)):

                midi_names[i] = os.path.join(root_folder, d, midi_names[i])
                try:
                    # we can transpose
                    multitrack = Multitrack(midi_names[i], beat_resolution=self.beat_resolution)
                    for j in range(-6, 6):
                        # need to augment data, so we can transpose the chords
                        self.midi_files.append((midi_names[i], j))

                except Exception as e:
                    print(e)
                    print(f"Couldn't read invalid file {midi_names[i]}")

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        # get multitrack
        multitrack = Multitrack(self.midi_files[idx][0], beat_resolution=self.beat_resolution)
        multitrack.transpose(self.midi_files[idx][1])
        # get piano roll of whole song
        pianoroll = multitrack.get_merged_pianoroll()
        one_hot_pianoroll = np.where(pianoroll!=0, 1, pianoroll)
        # shape is (num_time_steps, 128)
        # input is some music
        # output is music after?

        return torch.Tensor(one_hot_pianoroll.astype(np.float))


# note to self: 
# use embeddings
# using dense layer is crucial here to predict
# lstm layer is a feature layer, need dense layer for probability
# distribution
class MidiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MidiLSTM, self).__init__()
        # 2 stacked lstm
        # many folks use softmax to predict token, not whole vector
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2)
        self.fc = nn.Linear(hidden_dim, 128)

    def forward(self, x, hidden):
        x = self.lstm(x, hidden)
        x = self.fc(x[0])

        return x

# embed to EMBEDDING_DIM dimension vectors
# adding reduction="sum" will be make gradient updates more extreme
loss_function = nn.BCEWithLogitsLoss()
model = MidiLSTM(128, 256)
optimizer = optim.Adam(model.parameters(), lr=0.001)

md = MidiDataset("bach")

# input is (num of sequences, batch size, time step)
model.cuda()
for epoch in range(50):
    print("Epoch", epoch)
    vals = list(range(len(md)))
    np.random.shuffle(vals) 
    for i, val in enumerate(vals):
        hidden = (torch.randn(2, 1, 256).cuda(), torch.randn(2, 1, 256).cuda())
        # values in dataset
        optimizer.zero_grad()
        pianoroll = md[i][:-1]
        target = md[i][1:]


        x = pianoroll.view(-1, 1, 128).cuda()
        target = target.view(-1, 1, 128).cuda()
        output = model(x, hidden)
        loss = loss_function(output, target)
        loss.backward(retain_graph=True)
        if i % 10 == 0:
            print("Loss:", loss.item())
        optimizer.step()