import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from fantasy_football_fun.pfr.models import (
    AdvancedFantasyFootballLSTM,
    FantasyFootballLSTM
)

from fantasy_football_fun.pfr import FFDataset


class FantasyFootballTrainer():

    def __init__(self, dataset_path, normalize=True, train_test_split=0.9,
                 model_string="standard", loss_fnc="mse"):
        """
        creates the trainer.
        """
        self.dataset = FFDataset.FantasyFootballDataset(
                            dataset_path, normalize=normalize)
        data_len = len(self.dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        train_length = int(data_len * train_test_split)
        test_length = data_len - train_length
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [train_length, test_length])
        if model_string == "standard":
            self.model = FantasyFootballLSTM(1, 128)
        else:
            self.model = AdvancedFantasyFootballLSTM(1, 128)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if loss_fnc == "mse":
            self.loss_fnc = nn.MSELoss()
        else:
            self.loss_fnc = nn.L1Loss()

    def train(self, num_epochs):
        """
        trains the model.
        """
        total_epoch_train_losses = []
        total_epoch_test_losses = []
        for epoch in range(num_epochs):
            if epoch % 10 == 0:
                # save model every 10 epochs
                os.makedirs("models", exist_ok=True)
                torch.save(self.model.state_dict(), f"models/model_{epoch}.pth")
            losses = []
            print(f"NEW EPOCH {epoch}")
            vals = list(range(len(self.train_set)))
            np.random.shuffle(vals)
            for i, idx in enumerate(vals):
                hidden = (torch.randn(1, 1, 128).to(self.device),
                          torch.randn(1, 1, 128).to(self.device))
                self.optimizer.zero_grad()
                inputs = self.train_set[idx][0][:-1]
                target = self.train_set[idx][0][1:]
                if len(inputs) == 0:
                    continue
                # mean = torch.mean(inputs)
                # if mean <= 0.2:
                #     continue
                inputs = inputs.view(-1, 1, 1)
                target = target.view(-1, 1, 1)
                pos_onehot = self.train_set[idx][1]
                out = self.model(inputs, hidden, pos_onehot)
                loss = self.loss_fnc(out, target)
                losses.append(loss.item())
                loss.backward()
                if i % 100 == 0:
                    print("Loss:", loss.item())

                self.optimizer.step()
            mean_loss = np.mean(losses)

            total_epoch_train_losses.append(mean_loss)
            print(f"Mean TRAIN LOSS: {mean_loss}")

            # get testing losses
            test_losses = []
            for idx in range(len(self.test_set)):
                hidden = (torch.randn(1, 1, 128).cuda(),
                          torch.randn(1, 1, 128).cuda())
                inputs = self.test_set[idx][0][:-1]
                target = self.test_set[idx][0][1:]
                if len(inputs) == 0:
                    continue
                # mean = np.mean(inputs)
                # if mean <= 0.2:
                #     continue
                inputs = inputs.view(-1, 1, 1)
                target = target.view(-1, 1, 1)
                pos_onehot = self.train_set[idx][1]
                out = self.model(inputs, hidden, pos_onehot)
                loss = self.loss_fnc(out, target)
                test_losses.append(loss.item())

            test_mean = np.mean(test_losses)
            total_epoch_test_losses.append(test_mean)
            print(f'TEST LOSS: {test_mean}')
        np.savetxt('train_loss2.txt', np.array(total_epoch_train_losses))
        np.savetxt('test_loss2.txt', np.array(total_epoch_test_losses))


if __name__ == "__main__":
    trainer = FantasyFootballTrainer('finalized_players.csv')
    trainer.train(num_epochs=50)
