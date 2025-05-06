### diffusion_model/model.py
import torch
import torch.nn as nn

class MLPDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for timestep t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # predict noise in same shape
        )

    def forward(self, x, t):
        t = t.unsqueeze(-1).float() / 1000  # normalize timestep
        t_input = torch.cat([x, t], dim=1)
        return self.net(t_input)
