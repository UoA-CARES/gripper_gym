import logging

logging.basicConfig(level=logging.INFO)

import torch
import random
import numpy as np

from configurations import GripperEnvironmentConfig
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from gripper_trainer import GripperTrainer

from cares_reinforcement_learning.util import RLParser
from cares_reinforcement_learning.util import configurations as cares_cfg
import yaml


def main():
    parser = RLParser(GripperEnvironmentConfig)
    parser.add_configuration("gripper_config", GripperConfig)

    configurations = parser.parse_args()
    env_config: GripperEnvironmentConfig = configurations["env_config"]
    training_config: cares_cfg.TrainingConfig = configurations["training_config"]
    alg_config: cares_cfg.AlgorithmConfig = configurations["algorithm_config"]
    gripper_config: GripperConfig = configurations["gripper_config"]

    if env_config.is_debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"------------------------------------------")
    logging.info(
        f"\n**************\n"
        f"Environment Config:\n"
        f"**************\n"
        f"{yaml.dump(env_config.dict(), default_flow_style=False)}\n"
        f"\n**************\n"
        f"Algorithm Config:\n"
        f"**************\n"
        f"{yaml.dump(alg_config.dict(), default_flow_style=False)}"
        f"\n**************\n"
        f"Training Config:\n"
        f"**************\n"
        f"{yaml.dump(training_config.dict(), default_flow_style=False)}"
        f"\n**************\n"
        f"Gripper Config:\n"
        f"**************\n"
        f"{yaml.dump(gripper_config.dict(), default_flow_style=False)}"
    )
    logging.info(f"------------------------------------------")

    # input("Double check your experiment configurations :) Press ENTER to continue.")

    # TODO: reconcile the multiple seeds
    logging.info("Setting up Seeds")

    if len(training_config.seeds) > 1:
        logging.warning("Multiple seeds are not yet supported. Using the first seed.")

    # TODO replace with CARES_RL util for setting seeds
    torch.manual_seed(training_config.seeds[0])
    np.random.seed(training_config.seeds[0])
    random.seed(training_config.seeds[0])

    gripper_trainer = GripperTrainer(
        env_config=env_config,
        training_config=training_config,
        alg_config=alg_config,
        gripper_config=gripper_config,
    )

    gripper_trainer.train()


if __name__ == "__main__":
    main()
