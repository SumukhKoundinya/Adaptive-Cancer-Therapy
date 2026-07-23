import torch
import torch.nn as nn

class SurvivalModel(nn.Module):
    def __init__(self, input_dim):
        super(SurvivalModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class SurvivalTrainer:
    def __init__(self, input_dim, lr=1e-3):
        self.model = SurvivalModel(input_dim)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, x, y):
        pred = self.model(x)
        loss = self.loss_fn(pred.squeeze(), y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            return self.model(x)
