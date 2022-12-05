import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNetEncoder(nn.Module):
    def __init__(self, latent_dims=2, normalise=True):
        super().__init__()
        self.normalise = normalise
        self.latent_dims = latent_dims
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 10, 5, padding=2)
        self.conv3 = nn.Conv2d(10, 16, 5, padding=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.latent_dims)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.normalise:
            return x/x.norm(dim=1, keepdim=True)
        else:
            return x
