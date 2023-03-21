import logging
logging.basicConfig(level=logging.DEBUG)

import pydantic
from argparse import ArgumentParser

from pathlib import Path

import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from configurations import LearningConfig

from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer

from gripper_environment import GripperEnvironment

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logging.info("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    logging.info("Working with CPU")

def train(environment, network, memory):
    pass

def main():
    file_path = Path(__file__).parent.resolve()
    config = pydantic.parse_file_as(path=f"{file_path}/config/learning_config.json", type_=LearningConfig)
    environment = GripperEnvironment(config.env_config)

    # TODO take these from the environment via the Gripper setup
    MAX_ACTIONS = np.array([900, 750, 750, 900, 750, 750, 900, 750, 750]) #have generalised this to 750 for lower joints for consistency
    MIN_ACTIONS = np.array([100, 250, 250, 100, 250, 250, 100, 250, 250]) #have generalised this to 250 for lower joints for consistency
    action_num = 9
    observation_size = 10
    min_actions = MIN_ACTIONS
    max_actions = MAX_ACTIONS

    memory = MemoryBuffer(args.buffer_capacity)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    network = None
    # td3 = TD3(
    #     actor_network=actor,
    #     critic_one=critic_one,
    #     critic_two=critic_two,
    #     max_actions=max_actions,
    #     min_actions=min_actions,
    #     gamma=GAMMA,
    #     tau=TAU,
    #     device=DEVICE
    # )

    train(environment, network, memory)

if __name__ == '__main__':
    main()