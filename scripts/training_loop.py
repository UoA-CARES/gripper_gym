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

from configurations import LearningConfig, EnvironmentConfig, ObjectConfig
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from GripperTrainer import GripperTrainer

from cares_reinforcement_learning.util import Record, NetworkFactory

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--learning_config", type=str)
    parser.add_argument("--env_config",      type=str)
    parser.add_argument("--gripper_config",  type=str)
    parser.add_argument("--object_config",   type=str)
    parser.add_argument("--debug",      type=bool)

    home_path = os.path.expanduser('~')
    parser.add_argument("--local_results_path",  type=str, default=f"{home_path}/gripper_training")
    return parser.parse_args()

def main():

    args = parse_args()
    parent_path = Path(args.env_config).parent.absolute()

    env_config      = pydantic.parse_file_as(path=args.env_config,      type_=EnvironmentConfig)
    gripper_config  = pydantic.parse_file_as(path=args.gripper_config,  type_=GripperConfig)
    learning_config = pydantic.parse_file_as(path=args.learning_config, type_=LearningConfig)
    object_config   = pydantic.parse_file_as(path=args.object_config, type_=ObjectConfig)
    local_results_path = args.local_results_path
    is_debug = args.debug

    if is_debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Setting up Seeds")
    torch.manual_seed(learning_config.seed)
    np.random.seed(learning_config.seed)
    random.seed(learning_config.seed)

    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_path  = f"{date_time_str}_"
    file_path += f"RobotId{gripper_config.gripper_id}_EnvType{env_config.task}_ObsType{object_config.object_type}_Seed{learning_config.seed}_{learning_config.algorithm}"

    file_path = utils.create_directories(local_results_path, file_path)
    utils.store_configs(file_path, str(parent_path))

    gripper_trainer = GripperTrainer(env_config, gripper_config, learning_config, object_config, file_path)
    gripper_trainer.train()

if __name__ == '__main__':
    main()