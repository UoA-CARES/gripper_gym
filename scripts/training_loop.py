import logging
logging.basicConfig(level=logging.DEBUG)

from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

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

def parse_args():
    # TODO will replace this with yaml files to load in configurations for everything
    parser = ArgumentParser()
    file_path = Path(__file__).parent.resolve()

    parser.add_argument("--gripper_type",      type=int)
    parser.add_argument("--camera_id",         type=int, default=0)
    parser.add_argument("--marker_id",         type=int, default=0)
    parser.add_argument("--marker_size",       type=int, default=18)
    parser.add_argument("--camera_matrix",     type=str, default=f"{file_path}/config/camera_matrix.txt")
    parser.add_argument("--camera_distortion", type=str, default=f"{file_path}/config/camera_distortion.txt")

    parser.add_argument("--seed",              type=int, default=69)
    parser.add_argument("--batch_size",        type=int, default=32)
    parser.add_argument("--buffer_capacity",   type=int, default=100)
    parser.add_argument("--episode_num",       type=int, default=100)
    parser.add_argument("--action_num",        type=int, default=15)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    environment = GripperEnvironment(args.gripper_type,
                                     args.camera_id,
                                     args.marker_id,
                                     args.marker_size,
                                     args.camera_matrix,
                                     args.camera_distortion)

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