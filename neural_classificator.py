import torch.nn as nn


class NeuralClassificator(nn.Module):
    def __init__(self, buildWide: bool) -> None:
        super().__init__()
        self.layers = (
            nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32),nn.ReLU(), nn.Linear(32, 32),nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            if buildWide
            else nn.Sequential(
                nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
            )
        )

    def forward(self, x):
        return self.layers(x)
