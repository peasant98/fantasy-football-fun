import torch
from fantasy_football_fun.pfr.models import FantasyFootballLSTM
from fantasy_football_fun.pfr import FFDataset
import time
import numpy as np


class FantasyFootballPredictor():
    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FantasyFootballLSTM(1, 128)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, tup):
        hidden = (torch.randn(1, 1, 128).to(self.device),
                  torch.randn(1, 1, 128).to(self.device))
        x, pos = tup
        x = x.view(-1, 1, 1)
        outputs = self.model(x, hidden, pos)

        return outputs


if __name__ == '__main__':
    predictor = FantasyFootballPredictor("models/model_40.pth")
    dataset = FFDataset.FantasyFootballDataset(
                                'finalized_players.csv', normalize=True)
    scores = np.array([20, 20, 20, 20, 20, 20, 20])
    ppg_min = dataset.min
    ppg_max = dataset.max
    print(ppg_max)
    ppg_scale = ppg_max - ppg_min

    scores_norm = (scores - ppg_min) / (ppg_scale)
    scores_norm_tensor = torch.Tensor(scores_norm).to(predictor.device)
    pos_filler = torch.zeros(5).to(predictor.device)

    ppg_offset = ppg_min
    print(ppg_min)
    tup = (scores_norm_tensor, pos_filler)
    arr = predictor.predict(tup)
    ppg = (arr * ppg_scale) + ppg_offset
    print(ppg_min)
    careers = ppg.view(-1).detach().cpu().numpy()
    print(careers+3)