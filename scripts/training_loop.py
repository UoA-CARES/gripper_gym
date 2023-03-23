import logging
logging.basicConfig(level=logging.INFO)

import pydantic
from argparse import ArgumentParser

from pathlib import Path

import numpy as np
import random
import matplotlib.pyplot as plt

import torch

from configurations import LearningConfig, EnvironmentConfig, GripperConfig
from envrionments.RotationEnvironment import RotationEnvironment
from envrionments.TranslationEnvironment import TranslationEnvironment

from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer

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
    parser.add_argument("--env_config", type=str, default=f"{file_path}/config/env_4DOF_config.json")
    parser.add_argument("--gripper_config", type=str, default=f"{file_path}/config/gripper_4DOF_config.json")
    
    return parser.parse_args()

def main():
    args = parse_args()
    env_config      = pydantic.parse_file_as(path=args.env_config,      type_=EnvironmentConfig)
    gripper_config  = pydantic.parse_file_as(path=args.gripper_config,  type_=GripperConfig)
    learning_config = pydantic.parse_file_as(path=args.learning_config, type_=LearningConfig)

    if env_config.env_type == 0:
        environment = RotationEnvironment(env_config, gripper_config)
    elif env_config.env_type == 1:
        environment = TranslationEnvironment(env_config, gripper_config)
    
    logging.info("Resetting Environment")
    state = environment.reset()
    
    observation_size = len(state)
    num_actions = gripper_config.num_motors
    logging.info(f"Observaton Space: {observation_size} Action Space: {num_actions}")

    torch.manual_seed(learning_config.seed)
    np.random.seed(learning_config.seed)
    random.seed(learning_config.seed)

    logging.info("Setting up Network")
    # network = None

    logging.info("Setting up Memory")
    memory = MemoryBuffer(learning_config.buffer_capacity)

    logging.info("Starting Training Loop")
    # train(environment, network, memory)

if __name__ == '__main__':
    main()