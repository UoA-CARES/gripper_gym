import logging
import signal
import time
import random
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer

from gripper_environment import GripperEnvironment

import plotly.graph_objects as go
import pandas as pd


#TODO network setup should be shifted to its own module
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logging.info("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    logging.info("Working with CPU")

GAMMA = 0.995
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

#TODO should move this to a configuration file
MAX_ACTIONS = np.array([800, 750, 750, 800, 750, 750, 800, 750, 750]) #have generalised this to 750 for lower joints for consistency
MIN_ACTIONS = np.array([200, 250, 250, 200, 250, 250, 200, 250, 250]) #have generalised this to 250 for lower joints for consistency

#need to move these
class Actor(nn.Module):

    def __init__(self, observation_size, num_actions, learning_rate, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.hidden_size = [128, 64, 32]

        self.h_linear_1 = nn.Linear(in_features=observation_size, out_features=self.hidden_size[0])
        nn.ReLU()
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        nn.ReLU()
        nn.BatchNorm1d(self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        nn.ReLU()
        nn.BatchNorm1d(self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=num_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.sigmoid(self.h_linear_4(x))
        return x

class Critic(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Critic, self).__init__()

        self.hidden_size = [128, 64, 32]

        self.Q1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], self.hidden_size[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], 1)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.Q1(x)
        return q1

def ctrlc_handler(signum, frame):
    res = input("ctrl-c pressed. press anything to quit")
    exit()

def normalise_state(state):
    # modify normalisation
    normalise_state = []
    for i in range(0, len(state)-1):
        normalise_state.append(state[i]/1023)
    normalise_state.append(state[len(state)-1]/360)
    return normalise_state

#TODO - plot utils move to cares_lib or cares_reinforcement_learning
def plot_util():
    pass

#TODO: implement - likely best to move into networks
def save_models():
    pass

def train(args, network, memory: MemoryBuffer):    
    historical_reward = {}
    historical_reward["episode"] = []
    historical_reward["reward"] = []

    # df = pd.DataFrame(historical_reward)

    # scatter = go.Scatter()
    # figure  = go.FigureWidget(scatter)
    # figure.show()

    env = GripperEnvironment()
    
    for episode in range(0, args.episode_num):
        logging.info(f"Start of Episode {episode}")

        state = env.reset()
        normalised_state = normalise_state(state)
          
        episode_reward = 0
        step = 0
        done = False

        for step in range(0, args.number_steps):
            logging.info(f"Taking step {step}/{args.number_steps}")

            action = network.forward(normalised_state)            
            
            action = action.astype(int)
            next_state, reward, done, truncated = env.step(action)
            
            memory.add(state, action, reward, next_state, done)

            # TODO change to a percent of the max size or set size
            if len(memory.buffer) >= memory.buffer.maxlen:

                logging.info("Training Network")
                experiences = memory.sample(args.batch_size)
                for _ in range(0, args.G):
                    network.learn(experiences)

            state = next_state
            episode_reward += reward

            if done:
                logging.info("Episode {episode} was completed with {step} actions taken\n")
                break

        historical_reward["episode"].append(episode)
        historical_reward["reward"].append(episode_reward)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=6969)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--buffer_capacity", type=int, default=1000)
    parser.add_argument("--episode_num", type=int, default=1000)
    parser.add_argument("--number_steps", type=int, default=10)
    parser.add_argument("--G", type=int, default=10)
    return parser.parse_args()

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")    
    log_path = f"logs/{now}.log"
    logging.basicConfig(filename=log_path, level=logging.DEBUG)

    signal.signal(signal.SIGINT, ctrlc_handler)

    observation_size = 10
    action_size = 9

    args = parse_args()
    
    # TODO: change this once i change the max min thing in the servo class
    max_actions = MAX_ACTIONS
    min_actions = MIN_ACTIONS

    memory = MemoryBuffer(args.buffer_capacity)

    actor = Actor(observation_size, action_size, ACTOR_LR, max_actions)
    critic_one = Critic(observation_size, action_size, CRITIC_LR)
    critic_two = Critic(observation_size, action_size, CRITIC_LR)

    set_seeds(args.seed)

    td3 = TD3(
        actor_network=actor,
        critic_one=critic_one,
        critic_two=critic_two,
        max_actions=max_actions,
        min_actions=min_actions,
        gamma=GAMMA,
        tau=TAU,
        device=DEVICE
    )

    train(args, td3, memory)

if __name__ == '__main__':
    main()