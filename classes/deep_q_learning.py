import random
from collections import deque, namedtuple

import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple('Transition',
                        ('state', 'action', 'state_next', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_states, n_actions, hidden_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_states, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return self.layer3(x)