import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 300)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(300, 400)
        self.act2 = nn.ReLU()

        self.out = nn.Linear(400, 1)

    def init_weights(self):
        # Hidden layer 1
        fan_in1 = self.fc1.weight.size(1)
        bound1 = 1. / np.sqrt(fan_in1)
        nn.init.uniform_(self.fc1.weight, -bound1, bound1)
        nn.init.uniform_(self.fc1.bias, -bound1, bound1)

        # Hidden layer 2
        fan_in2 = self.fc2.weight.size(1)
        bound2 = 1. / np.sqrt(fan_in2)
        nn.init.uniform_(self.fc2.weight, -bound2, bound2)
        nn.init.uniform_(self.fc2.bias, -bound2, bound2)

        # Output layer (Q-value)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.out(x)
        return x
