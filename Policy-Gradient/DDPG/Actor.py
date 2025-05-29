import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 300)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(300, 400)
        self.act2 = nn.ReLU()
        
        self.out = nn.Linear(400, action_dim)
        self.act3 = nn.Tanh()

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

        # Output layer (action)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state):
        x = self.act1(self.fc1(state))
        x = self.act2(self.fc2(x))
        x = self.act3(self.out(x))
        return x
