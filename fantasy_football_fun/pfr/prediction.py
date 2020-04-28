import torch
from fantasy_football_fun.pfr.models import ModifiedFantasyFootballLSTM


class FantasyFootballPredictor():
    def __init__(self, model_path):
        self.model = ModifiedFantasyFootballLSTM(1, 128)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.cuda()

    def predict(self, x):
        outputs = self.model(x)

        print(outputs)


if __name__ == '__main__':
    predictor = FantasyFootballPredictor("models/eighthmodel_45.pth")