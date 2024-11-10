import logging

logging.basicConfig(level=logging.INFO)

import os
import pydantic
import torch
import random
import json
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


# def parse_args():
#     parser = ArgumentParser()

#     parser.add_argument("--env_config", type=str)
#     parser.add_argument("--training_config", type=str)
#     parser.add_argument("--algorithm_config", type=str)
#     parser.add_argument("--gripper_config", type=str)

#     return parser.parse_args()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config_folder", type=str, required=True, help="Path to the folder containing all the configuration files")
    return parser.parse_args()

def find_config_files(config_folder):
    # Find all the configuration files in the folder
    config_folder = Path(config_folder)

    env_config_path = config_folder / "env_config.json"
    training_config_path = config_folder / "training_config.json"
    algorithm_config_path = config_folder / "alg_config.json"
    gripper_config_path = config_folder / "gripper_config.json"

    if not env_config_path.exists():
        raise FileNotFoundError(f"Environment configuration file not found at {env_config_path}")
    if not training_config_path.exists():
        raise FileNotFoundError(f"Training configuration file not found at {training_config_path}")
    if not algorithm_config_path.exists():
        raise FileNotFoundError(f"Algorithm configuration file not found at {algorithm_config_path}")
    if not gripper_config_path.exists():
        raise FileNotFoundError(f"Gripper configuration file not found at {gripper_config_path}")
    
    return {
        "env_config": env_config_path,
        "training_config": training_config_path,
        "algorithm_config": algorithm_config_path,
        "gripper_config": gripper_config_path
    }

def load_config(config_path):
    configuration = {}

    with open(config_path["env_config"], "r") as env_file:
        configuration["env_config"] = json.load(env_file)
    with open(config_path["training_config"], "r") as training_file:
        configuration["training_config"] = json.load(training_file)
    with open(config_path["algorithm_config"], "r") as algorithm_file:
        configuration["algorithm_config"] = json.load(algorithm_file)
    with open(config_path["gripper_config"], "r") as gripper_file:
        configuration["gripper_config"] = json.load(gripper_file)

    return configuration


def main():
    # Parse arguments
    args = parse_args()

    #Locate conifg files
    config_paths = find_config_files(args.config_folder)

    parser = RLParser(GripperEnvironmentConfig)
    parser.add_configuration("gripper_config", GripperConfig)

    # configurations = parser.parse_args()
    config_dicts = load_config(config_paths)
    configurations = {
        "env_config": GripperEnvironmentConfig(**config_dicts["env_config"]),
        "training_config": cares_cfg.TrainingConfig(**config_dicts["training_config"]),
        "algorithm_config": cares_cfg.AlgorithmConfig(**config_dicts["algorithm_config"]),
        "gripper_config": GripperConfig(**config_dicts["gripper_config"])
    }

    print(configurations)
    env_config: GripperEnvironmentConfig = configurations["env_config"]
    training_config: cares_cfg.TrainingConfig = configurations["training_config"]
    alg_config: cares_cfg.AlgorithmConfig = configurations["algorithm_config"]
    gripper_config: GripperConfig = configurations["gripper_config"]
    print(f"------------------------------------------")
    print(f"env_config: {env_config}")
    print(f"training_config: {training_config}")
    print(f"alg_config: {alg_config}")
    print(f"gripper_config: {gripper_config}")

    # Set seeds
    logging.info("Setting up Seeds")
    torch.manual_seed(training_config.seeds[0])
    np.random.seed(training_config.seeds[0])
    random.seed(training_config.seeds[0])

    # Initialize Trainer
    gripper_trainer = GripperTrainer(
        env_config, training_config, alg_config, gripper_config
    )

    print(gripper_trainer)

    # # Load Models and just evaluate
    # file_path = input("Enter the main folder file path: ")
    # model_name = input("Enter the model name (alg-checkpoint-num): ")
    # gripper_trainer.agent.load_models(file_path, model_name)
    # print('Successfully Loaded models')
    
    # gripper_trainer.evaluation_loop(total_steps = 0, num_eval_steps = 100, num_eval_episodes = 20)


if __name__ == "__main__":
    main()