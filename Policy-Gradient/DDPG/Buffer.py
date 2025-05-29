import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5), device='cuda'):
        self.device = device
        self.s = torch.zeros((max_size, *state_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((max_size, *action_dim), dtype=torch.long, device=device)
        self.r = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.s_prime = torch.zeros((max_size, *state_dim), dtype=torch.float32, device=device)
        self.terminated = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

        self.ptr, self.size, self.max_size = 0, 0, max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = torch.tensor(s, device=self.device)
        self.a[self.ptr] = torch.tensor(a, device=self.device)
        self.r[self.ptr] = torch.tensor(r, device=self.device)
        self.s_prime[self.ptr] = torch.tensor(s_prime, device=self.device)
        self.terminated[self.ptr] = torch.tensor(terminated, device=self.device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.s[ind], self.a[ind], self.r[ind], self.s_prime[ind], self.terminated[ind]
