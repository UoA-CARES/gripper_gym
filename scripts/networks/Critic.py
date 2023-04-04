import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from networks.weight_bias_init import weight_init


class Critic(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Critic, self).__init__()

        self.hidden_size = [1024, 512, 256, 128]

        # Q1 architecture
        self.h_linear_1 = nn.Linear(in_features=observation_size + num_actions, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=self.hidden_size[3])
        self.h_linear_5 = nn.Linear(in_features=self.hidden_size[3], out_features=1)

        # Q2 architecture
        self.h_linear_1_2 = nn.Linear(in_features=observation_size + num_actions, out_features=self.hidden_size[0])
        self.h_linear_2_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_2_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_2_4 = nn.Linear(in_features=self.hidden_size[2], out_features=self.hidden_size[3])
        self.h_linear_2_5 = nn.Linear(in_features=self.hidden_size[3], out_features=1)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(weight_init)

    def forward(self, state, action):
        obs_action = torch.cat([state, action], dim=1)

        q1 = F.relu(self.h_linear_1(obs_action))
        q1 = F.relu(self.h_linear_2(q1))
        q1 = F.relu(self.h_linear_3(q1))
        q1 = F.relu(self.h_linear_4(q1))
        q1 = self.h_linear_5(q1)

        q2 = F.relu(self.h_linear_1_2(obs_action))
        q2 = F.relu(self.h_linear_2_2(q2))
        q2 = F.relu(self.h_linear_2_3(q2))
        q2 = F.relu(self.h_linear_2_4(q2))
        q2 = self.h_linear_2_5(q2)

        return q1, q2
