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

from configurations import LearningConfig, EnvironmentConfig, GripperConfig

from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer

from envrionments.RotationEnvironment import RotationEnvironment

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logging.info("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    logging.info("Working with CPU")

def train(environment, network, memory):
    pass

def parse_args():
    parser = ArgumentParser()
    file_path = Path(__file__).parent.resolve()
    
    parser.add_argument("--learning_config", type=str, default=f"{file_path}/config/learning_config.json")
    parser.add_argument("--env_config", type=str, default=f"{file_path}/config/env_config.json")
    parser.add_argument("--gripper_config", type=str, default=f"{file_path}/config/gripper_config.json")
    
    return parser.parse_args()

def main():
    args = parse_args()
    env_config      = pydantic.parse_file_as(path=args.env_config,      type_=EnvironmentConfig)
    gripper_config  = pydantic.parse_file_as(path=args.gripper_config,  type_=GripperConfig)
    learning_config = pydantic.parse_file_as(path=args.learning_config, type_=LearningConfig)

    environment = RotationEnvironment(env_config, gripper_config)


    # read out all the learning configurations that are required
    
    # num_actions = learning_config.num_actions
    # observation_size = learning_config.observation_space

    # memory = MemoryBuffer(args.buffer_capacity)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    # network = None
    
    # train(environment, network, memory)

if __name__ == '__main__':
    main()