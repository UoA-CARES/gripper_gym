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

from configurations import LearningConfig, GripperEnvironmentConfig, ObjectConfig
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from GripperTrainer import GripperTrainer

from cares_reinforcement_learning.util import Record, NetworkFactory, RLParser


def main():
    parser = RLParser(GripperEnvironmentConfig)
    parser.add_configuration("gripper_config", GripperConfig)
    parser.add_configuration("object_config", ObjectConfig)

    configurations = parser.parse_args()
    env_config = configurations["env_config"] 
    training_config = configurations["training_config"]
    alg_config = configurations["algorithm_config"]
    gripper_config = configurations["gripper_config"]
    object_config = configurations["object_config"]

    if env_config.is_debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Setting up Seeds")
    torch.manual_seed(learning_config.seed)
    np.random.seed(learning_config.seed)
    random.seed(learning_config.seed)

    # Replace with Record
    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_path  = f"{date_time_str}_"
    file_path += f"RobotId{gripper_config.gripper_id}_EnvType{env_config.task}_ObsType{object_config.object_type}_Seed{learning_config.seed}_{learning_config.algorithm}"

    file_path = utils.create_directories(local_results_path, file_path)
    utils.store_configs(file_path, str(parent_path))

    gripper_trainer = GripperTrainer(env_config, gripper_config, training_config, object_config, file_path)
    gripper_trainer.train()

if __name__ == '__main__':
    main()