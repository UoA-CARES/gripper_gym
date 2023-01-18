
"""
This is an example script that shows how one uses the cares reinforcement learning package.
To run this specific example, move the file so that it is at the same level as the package root
directory
    -- script.py
    -- summer_reinforcement_learning/
"""

#network
#memory replays 

#TODO: track the error messages so i know when things are breaking
from datetime import datetime

from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer
#TODO: figure out why this isnt working
#from cares_reinforcement_learning.examples.Actor import Actor
#from cares_reinforcement_learning.examples.Critic import Critic

from gripper_environment import Environment
import numpy as np
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt

#for ctrl-c handling
import signal
import time

#these are just for the networks that should get moved
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")

GAMMA = 0.995
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

MAX_ACTIONS = np.array([800, 750, 750, 800, 750, 750, 800, 750, 750])  #have generalised this to 750 for lower joints for consistency
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
    res = input("ctrl-c pressed. press y to exit and save")
    if res == 'y':
        plt.show()
        plt.savefig('terminatedplt.png')
        exit()

def normalise_state(state):
    # modify normalisation
    normalise_state = []
    for i in range(0, len(state-1)):
        normalise_state[i] = state[i]/1023
    normalise_state[i+1] = state[i+1]/360
    return normalise_state

def train(network, memory: MemoryBuffer):

    args = parse_args()

    now = datetime.now()
    now = now.strftime("%Y-%M-%D-%H-%M-%S")
    logger = open(f"logs/{now}-log.txt", "a")
    
    historical_reward = []

    env = Environment()

    for episode in range(0, args.episode_num):
        print(f"Start of Episode {episode}")

        state, terminated = env.reset()
        if terminated:
            print(f"Gripper failed to go to initial home on episode {episode}/{args.episode_num}!")
            exit()

        normalised_state = normalise_state(state)
          
        episode_reward = 0
        step = 0
        terminated = False

        for step in range(0, args.number_steps):
            print(f"Taking step {step}/{args.number_steps}")

            action = network.forward(state)
            
            action = action.astype(int)
            next_state, reward, terminated, truncated = env.step(action)
            
            memory.add(state, action, reward, next_state, terminated)

            if len(memory.buffer) >= memory.buffer.maxlen:
                experiences = memory.sample(args.batch_size)
                
                for _ in range(0, args.G):
                    network.learn(experiences)

            state = next_state
            episode_reward += reward

            if terminated:
                print("Episode Terminated")
                logger.write(f"The current epsiode is {episode}, this was TERMINATED at {step} actions taken\n")
                break

        historical_reward.append(episode_reward)
    
    # plt.xlabel("Episode")
    # plt.ylabel("Reward") 
    # plt.title("Reward per Episode")     
    # xint = []
    # locs, labels = plt.xticks()
    # for each in locs:
    #     xint.append(int(each))
    # plt.xticks(xint)
    # plt.show()
    # plt.savefig('testing181_2.png')

    # logger.write("\n")
    # logger.close

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=6969)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--buffer_capacity", type=int, default=1000)
    parser.add_argument("--episode_num", type=int, default=1000)
    parser.add_argument("--number_steps", type=int, default=10)
    parser.add_argument("--G", type=int, default=10)

    args = parser.parse_args()
    return args

def main():

    signal.signal(signal.SIGINT, ctrlc_handler)

    observation_size = 10
    action_size = 9

    #setup the grippers
    args = parse_args()
    
    # TODO: change this once i change the max min thing in the servo class
    max_actions = MAX_ACTIONS
    min_actions = MIN_ACTIONS

    memory = MemoryBuffer(args.buffer_capacity)

    actor = Actor(observation_size, action_size, ACTOR_LR, max_actions)
    critic_one = Critic(observation_size, action_size, CRITIC_LR)
    critic_two = Critic(observation_size, action_size, CRITIC_LR)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    print(f"Filling Buffer...")

    train(td3, memory)


if __name__ == '__main__':
    main()