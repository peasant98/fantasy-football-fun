import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class FantasyFootballDataset(Dataset):
    """
    dataset for fantasy football data.
    """
    def __init__(self, csv, normalize=True, optimize_separately=True, verbose=False):
        # get the list of directories
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # over 30k entries long of player seasons
        df = pd.read_csv(csv)
        # get player names
        self.player_names = np.unique(df['Player'].values)
        self.correct_positions = ['QB', 'RB', 'WR', 'TE', 'K']
        correct_df = df[df['position'].isin(self.correct_positions)]
        sorted_df = correct_df.sort_values(['Player', 'Year'])
        self.df = sorted_df.dropna()
        ppg = 'FantPtPerGame'
        self.verbose = verbose
        self.pos_mapping = {}
        self.optimize_separately = optimize_separately
        self.player_names = np.unique(self.df['Player'].values)

        for idx, position in enumerate(self.correct_positions):
            self.pos_mapping[position] = idx

        if normalize:
            cols_to_norm = [ppg]
            self.df[cols_to_norm] = self.df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    def __len__(self):
        return len(self.player_names)

    def __getitem__(self, idx):
        player_name = self.player_names[idx]
        player_seasons_df = self.df.loc[self.df['Player'] == player_name]
        res = []
        nums = []
        for index, row in player_seasons_df.iterrows():
            one_hot = np.zeros(len(self.correct_positions))
            position = row['position']
            nums.append(row['FantPtPerGame'])
            one_hot[self.pos_mapping[position]] = 1.0
            res.append(row['FantPtPerGame'])
        if self.verbose:
            plt.plot(nums, c='green')
            plt.title(f'{position}: {player_name}')
            plt.show()
        """
        data needed for net:
        position, name, games FantPtPerGame

        """
        if self.optimize_separately:

            return torch.Tensor(res).to(self.device), torch.Tensor(one_hot).to(self.device)
        else:
            res.extend(one_hot)
            return torch.Tensor(res).to(self.device)
