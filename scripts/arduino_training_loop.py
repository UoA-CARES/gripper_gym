from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer

from gripper_environment import GripperEnvironment
import numpy as np
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt

# DOESNT THROW ERROR WHEN HOME CANNOT BE REACHED
# Servo 3 stuck at same position a lot of the time


import torch
import torch.nn as nn
import torch.optim as optim

import logging

logging.basicConfig(level=logging.DEBUG)

# if torch.cuda.is_available():
#    DEVICE = torch.device('cuda')
# else:
DEVICE = torch.device("cpu")

GAMMA = 0.995
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

MAX_ACTIONS = np.array(
    [900, 750, 750, 900, 750, 750, 900, 750, 750]
)  # have generalised this to 750 for lower joints for consistency
MIN_ACTIONS = np.array(
    [100, 250, 250, 100, 250, 250, 100, 250, 250]
)  # have generalised this to 250 for lower joints for consistency

env = GripperEnvironment()


class Actor(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.hidden_size = [128, 64, 32]

        self.h_linear_1 = nn.Linear(
            in_features=observation_size, out_features=self.hidden_size[0]
        )
        self.h_linear_2 = nn.Linear(
            in_features=self.hidden_size[0], out_features=self.hidden_size[1]
        )
        self.h_linear_3 = nn.Linear(
            in_features=self.hidden_size[1], out_features=self.hidden_size[2]
        )
        self.h_linear_4 = nn.Linear(
            in_features=self.hidden_size[2], out_features=num_actions
        )

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
            nn.Linear(self.hidden_size[2], 1),
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.Q1(x)
        return q1


def main():

    observation_size = 10

    action_num = 9

    # setup the grippers
    args = parse_args()

    # TODO: change this once i change the max min thing in the servo class
    max_actions = MAX_ACTIONS
    min_actions = MIN_ACTIONS

    memory = MemoryBuffer(args.buffer_capacity)

    actor = Actor(observation_size, action_num, ACTOR_LR, max_actions)
    critic_one = Critic(observation_size, action_num, CRITIC_LR)
    critic_two = Critic(observation_size, action_num, CRITIC_LR)

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
        device=DEVICE,
    )

    print(f"Filling Buffer...")

    fill_buffer(memory)

    train(td3, memory)


def train(td3, memory: MemoryBuffer):

    args = parse_args()

    historical_reward = []

    state = env.gripper.home()
    # to get the array the correct length for the first action I need to
    state.append(-1)

    for episode in range(0, args.episode_num):

        env.gripper.home()

        # map state values to 0 - 1
        for i in range(0, len(state)):
            state[i] = (state[i]) / 360

        episode_reward = 0
        Done = False
        action_taken = 0

        print(f"episode {episode}")
        # print(state)

        while not Done and action_taken < args.action_num:

            # Select an Action
            # td3.actor_net.eval() --> dont need bc we are not using batch norm???
            with torch.no_grad():

                state_tensor = torch.FloatTensor(state)

                state_tensor = state_tensor.to(DEVICE)
                action = td3.forward(state_tensor)  # potientially a naming conflict
                # action = action.numpy()

            td3.actor_net.train(True)

            # convert actor output to valid integer steps within the max and min
            for i in range(0, len(action)):
                # map 0 - 1 to min - max
                action[i] = (action[i]) * (
                    MAX_ACTIONS[i] - MIN_ACTIONS[i]
                ) + MIN_ACTIONS[i]

            # make sure the actions are int
            action = action.astype(int)

            next_state, reward, terminated, Done = env.step(action)

            memory.add(state, action, reward, next_state, Done)

            for _ in range(0, 10):  # can be bigger
                experiences = memory.sample(args.batch_size)
                td3.learn(experiences)

            action_taken += 1
            print(f"actions taken = {action_taken}")

            state = next_state
            episode_reward += reward

            # im not too sure whether this is working properly
            if terminated:
                print("Episode Terminated")
                historical_reward.append(episode_reward)
                plt.plot(historical_reward)
                episode += 1
                # episode terminated, therefore break out of while loop
                env.gripper.home()
                Done = True

        historical_reward.append(episode_reward)
        print(f"Episode #{episode} Reward {episode_reward}")

        plt.plot(historical_reward)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    xint = []
    locs, labels = plt.xticks()
    for each in locs:
        xint.append(int(each))
    plt.xticks(xint)
    plt.show()

    torch.save(td3, "/home/anyone/gripperCode/Gripper-Code/models/trained_model.pt")


def fill_buffer(memory):

    env.gripper.ping()
    state = env.reset()

    while len(memory.buffer) < memory.buffer.maxlen:

        # TODO: refactor the code surely i can make it better than this
        action = np.zeros(9)
        action_taken = 0  # need it for step but is irrevilant at this point

        for i in range(0, len(MAX_ACTIONS)):
            action[i] = np.random.randint(MIN_ACTIONS[i], MAX_ACTIONS[i])

        action = action.astype(int)

        # pick a random target angle
        target_angle = np.random.randint(0, 360)
        # TODO: would be good to have a thing here to add a thing to the memory if the actions terminated

        next_state, reward, terminated, truncated = env.step(action)
        print(
            f"State: {state} \nAction: {action} \nReward: {reward} \nNext State: {next_state} \nTerminated{terminated} \nTruncated: {truncated}"
        )
        # print(reward)
        memory.add(state, action, reward, next_state, terminated)
        # keep track of how full the buffer is
        print(f"Buffer: {len(memory.buffer)} / {memory.buffer.maxlen}", end="\r")
        state = next_state


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_capacity", type=int, default=1200)
    parser.add_argument("--episode_num", type=int, default=1000)
    parser.add_argument("--action_num", type=int, default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()