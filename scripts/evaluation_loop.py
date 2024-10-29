import logging

logging.basicConfig(level=logging.INFO)

import os
import pydantic
import torch
import random
import numpy as np
import tools.utils as utils

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from cares_reinforcement_learning.util import RLParser
from configurations import GripperEnvironmentConfig
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_reinforcement_learning.util import configurations as cares_cfg

from gripper_trainer import GripperTrainer


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--env_config", type=str)
    parser.add_argument("--training_config", type=str)
    parser.add_argument("--algorithm_config", type=str)
    parser.add_argument("--gripper_config", type=str)

    return parser.parse_args()


def main():

    parser = RLParser(GripperEnvironmentConfig)
    parser.add_configuration("gripper_config", GripperConfig)

    configurations = parser.parse_args()
    env_config: GripperEnvironmentConfig = configurations["env_config"]
    training_config: cares_cfg.TrainingConfig = configurations["training_config"]
    alg_config: cares_cfg.AlgorithmConfig = configurations["algorithm_config"]
    gripper_config: GripperConfig = configurations["gripper_config"]

    logging.info("Setting up Seeds")
    torch.manual_seed(training_config.seeds[0])
    np.random.seed(training_config.seeds[0])
    random.seed(training_config.seeds[0])

    gripper_trainer = GripperTrainer(
        env_config, training_config, alg_config, gripper_config
    )
    # Load Models and just evaluate
    file_path = input("Enter the main folder file path: ")
    model_name = input("Enter the model name (alg-checkpoint-num): ")
    gripper_trainer.agent.load_models(file_path, model_name)
    print('Successfully Loaded models')
    
    gripper_trainer.evaluation_loop(total_steps = 0, num_eval_steps = 5, num_eval_episodes = 5)


if __name__ == "__main__":
    main()