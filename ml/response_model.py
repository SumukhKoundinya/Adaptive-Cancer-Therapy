import torch
import torch.nn as nn

class ResponseModel(nn.Module):
    def __init__(self, input_dim):
        super(ResponseModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)

class ResponseTrainer:
    def __init__(self, input_dim, lr=1e-3):
        self.model = ResponseModel(input_dim)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, x, y):
        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.model(x)
            return torch.argmax(logits, dim=1)
