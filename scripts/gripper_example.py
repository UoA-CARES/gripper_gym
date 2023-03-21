import logging
logging.basicConfig(level=logging.DEBUG)

import argparse
import pydantic

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from gripper_environment import EnvironmentConfig
import grippers.gripper_helper as ghlp

# Example of how to use Gripper
def main(config):
    logging.info(f"Running gripper {config.gripper_config.gripper_type}")

    gripper = ghlp.create_gripper(config.gripper_config)

    logging.info("Pinging Gripper to find all servos")
    gripper.ping()

    logging.info("Moving the Gripper to a home position")
    gripper.home()

    logging.info("Closing the Gripper")
    gripper.close()

if __name__ == "__main__":
    config = pydantic.parse_file_as(path=f"{file_path}/config/environment_config.json", type_=EnvironmentConfig)
    logging.info(f"Config: {config}")
    main(config)
