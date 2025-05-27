import numpy as np
import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelCNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(DuelCNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        
        # dueling architecture
        self.fc_value = nn.Linear(256, 1)
        self.fc_advantage = nn.Linear(256, action_dim)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.view(-1, self.in_features)
        x = self.activation(self.fc1(x))
        
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)

        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

class DQN:
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=0.00025,
            epsilon=1.0,
            epsilon_min=0.1,
            gamma=0.99,
            batch_size=32,
            warmup_steps=5000,
            buffer_size=int(1e5),
            target_update_interval=10000,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = DuelCNNActionValue(state_dim[0], action_dim)
        self.target_network = DuelCNNActionValue(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)
        self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0

        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_rate = np.log(self.epsilon_start / epsilon_min) / 2e6

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)

        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)

        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()

        return a

    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))

        next_q_online = self.network(s_prime).detach()
        max_action = next_q_online.argmax(dim=1, keepdim=True)
        
        next_q_target = self.target_network(s_prime).detach()
        next_q = next_q_target.gather(1, max_action)

        td_target = r + (1. - terminated) * self.gamma * next_q
        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }

        return result

    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * np.exp(-self.epsilon_rate * self.total_steps)
        self.epsilon = max(self.epsilon_min, self.epsilon)
        return result


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s 
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]),
        )
