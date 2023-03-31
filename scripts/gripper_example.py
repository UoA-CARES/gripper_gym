import logging
logging.basicConfig(level=logging.INFO)
import pydantic

from pathlib import Path
file_path = Path(__file__).parent.resolve()

import time

from Gripper import Gripper
from configurations import GripperConfig

# Example of how to use Gripper
def main(gripper_config):
    logging.info(f"Running gripper {gripper_config.gripper_type}")

    gripper = Gripper(gripper_config)

    logging.info("Pinging Gripper to find all servos")
    gripper.ping()

    logging.info("Moving the Gripper to a home position")
    gripper.home()

    logging.info("Gripper State")
    gripper_state = gripper.state()
    logging.info(gripper_state)

    velocities = [-30,0,0,-50,0,0,-30,0,0]
    logging.info(f"Set Velocity: {velocities}")
    gripper.move_velocity(velocities) 

    start_time = time.perf_counter()
    while time.perf_counter() < start_time + 10:
        gripper.step()
        time.sleep(0.1)

    velocities = [30,0,0,50,0,0,30,0,0]
    logging.info(f"Set Velocity: {velocities}")
    gripper.move_velocity(velocities) 

    start_time = time.perf_counter()
    while time.perf_counter() < start_time + 3:
        gripper.step()
        time.sleep(0.1)

    logging.info(f"Setting velocity to zero")
    gripper.move_velocity([0,0,0,0,0,0,0,0,0])
        
    start_time = time.perf_counter()
    while time.perf_counter() < start_time + 2:
        gripper.step()
        time.sleep(0.1)

    logging.info("Moving the Gripper to a home position")
    gripper.home()

    time.sleep(1.0)

    logging.info("Gripper State")
    gripper_state = gripper.state()
    logging.info(gripper_state)

    logging.info("Closing the Gripper")
    gripper.close()


if __name__ == "__main__":
    config = pydantic.parse_file_as(path=f"{file_path}/config/gripper_9DOF_config.json", type_=GripperConfig)
    logging.info(f"Config: {config}")
    main(config)
