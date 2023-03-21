import logging
logging.basicConfig(level=logging.DEBUG)
import pydantic

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from configurations import LearningConfig
import grippers.gripper_helper as ghlp

# Example of how to use Gripper
def main(config):
    logging.info(f"Running gripper {config.env_config.gripper_config.gripper_type}")

    gripper = ghlp.create_gripper(config.env_config.gripper_config)

    logging.info("Pinging Gripper to find all servos")
    gripper.ping()

    logging.info("Moving the Gripper to a home position")
    gripper.home()

    logging.info("Closing the Gripper")
    gripper.close()

if __name__ == "__main__":
    config = pydantic.parse_file_as(path=f"{file_path}/config/learning_config.json", type_=LearningConfig)
    logging.info(f"Config: {config}")
    main(config)
