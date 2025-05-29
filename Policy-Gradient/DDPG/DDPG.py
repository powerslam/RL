import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from Actor import Actor, OUNoise
from Critic import Critic

from Buffer import ReplayBuffer

class DDPG:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr * 10)

        self.gamma = gamma
        self.tau = tau

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size=int(1e6))

        self.batch_size = 64

        self.ou_noise = OUNoise(action_dim)

    def learn(self):
        s, a, r, s_prime, done = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        
        # target network는 soft update를 통해 업데이트되므로, target network의 파라미터를 고정해야 한다.
        with torch.no_grad():
            next_q_target = r + (1. - done) * self.gamma * self.target_critic(s_prime, self.target_actor(s_prime))

        # critic loss는 MSE로 계산한다.
        critic_loss = F.mse_loss(self.critic(s, a), next_q_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor loss는 critic의 Q값을 최대화하는 방향으로 업데이트한다.
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target network 업데이트 == soft update
        for target_param, source_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    @torch.no_grad()
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().numpy()[0]

        noise = self.ou_noise.sample()
        action = np.clip(action + noise, -1, 1)

        return action

    def process(self, state, action, reward, next_state, done):
        self.buffer.update(state, action, reward, next_state, done)
        
        if len(self.buffer) >= self.batch_size:
            self.learn()

        result = {
            'actor_loss': self.actor_optimizer.param_groups[0]['lr'],
            'critic_loss': self.critic_optimizer.param_groups[0]['lr']
        }

        return result        
