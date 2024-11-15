import logging

logging.basicConfig(level=logging.INFO)

import torch
import random
import numpy as np
from datetime import datetime


from configurations import GripperEnvironmentConfig
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from gripper_trainer import GripperTrainer

from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util import RLParser
from cares_reinforcement_learning.util.rl_parser import RunConfig
from cares_reinforcement_learning.util import configurations as cares_cfg
from cares_reinforcement_learning.util import helpers as hlp
import yaml


def main():
    parser = RLParser(GripperEnvironmentConfig)
    parser.add_configuration("gripper_config", GripperConfig)

    configurations = parser.parse_args()
    run_config: RunConfig = configurations["run_config"]
    env_config: GripperEnvironmentConfig = configurations["env_config"]
    training_config: cares_cfg.TrainingConfig = configurations["train_config"]
    alg_config: cares_cfg.AlgorithmConfig = configurations["alg_config"]
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

    run_name = input(
        "Double check your experiment configurations :) Press ENTER to continue. (Optional - Enter a name for this run)\n"
    )

    logging.info(f"Command: {run_config.command}")

    # file_path = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-gripper{gripper_config.gripper_id}-{env_config.task}-{alg_config.algorithm}-{training_config.seeds}-{gripper_config.action_type}'

    format_str="{algorithm}/{algorithm}-{gripper_id}-{domain_task}-{run_name}{date}"

    base_log_dir = Record.create_base_directory(
        domain=env_config.domain,
        task=env_config.task,
        gym="gripper_gym",
        algorithm=alg_config.algorithm,
        run_name=run_name,
        gripper_id=gripper_config.gripper_id,
        format_str=format_str,
    )

    record = Record(
        base_directory=base_log_dir,
        task=env_config.task,
        algorithm=alg_config.algorithm,
        agent=None,
    )

    record.save_configurations(configurations)

    for iteration, seed in enumerate(training_config.seeds):
        logging.info(f"Iteration {iteration + 1}/{len(training_config.seeds)} with seed {seed}")

        # TODO replace with CARES_RL util for setting seeds
        logging.info("Setting up Seeds")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        gripper_trainer = GripperTrainer(
            env_config=env_config,
            training_config=training_config,
            alg_config=alg_config,
            gripper_config=gripper_config,
            record=record,
        )

        record.set_sub_directory(seed)
        record.set_agent(gripper_trainer.agent)

        # TODO add a check for train vs evaluation command
        if run_config.command == "train":
            gripper_trainer.train()
        elif run_config.command == "evaluate":
            raise NotImplementedError("Evaluation is not yet implemented")

        # TODO Keep this but just only pass one seed if you only want to handle one seed
        if iteration > 0:
            logging.warning("Multiple seeds are not yet supported - aborting after first iteration.")
            break

if __name__ == "__main__":
    main()
