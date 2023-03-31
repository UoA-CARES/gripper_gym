import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from networks.weight_bias_init import weight_init


class Actor(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Actor, self).__init__()

        self.hidden_size = [1024, 512, 256, 128]

        self.h_linear_1 = nn.Linear(in_features=observation_size,    out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=self.hidden_size[3])
        self.h_linear_5 = nn.Linear(in_features=self.hidden_size[3], out_features=num_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(weight_init)

    def forward(self, state):
        x = F.relu(self.h_linear_1(state))
        x = F.relu(self.h_linear_2(x))
        x = F.relu(self.h_linear_3(x))
        x = F.relu(self.h_linear_4(x))
        x = torch.tanh(self.h_linear_5(x))
        return x
