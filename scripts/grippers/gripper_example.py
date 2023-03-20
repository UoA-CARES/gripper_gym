import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

from ArduinoGripper import ArduinoGripper
from U2D2Gripper import U2D2Gripper

# Example of how to use Gripper
def main(gripper_type):
    logging.info(f"Running gripper {gripper_type}")

    gripper = None
    if gripper_type == 0:# U2D2
        gripper = U2D2Gripper()
    elif gripper_type == 1:# Arduino
        gripper = ArduinoGripper()

    logging.info("Pinging Gripper to find all servos")
    gripper.ping()

    logging.info("Moving the Gripper to a home position")
    gripper.home()

    logging.info("Closing the Gripper")
    gripper.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('gripper_type', type=int, help='Require defining which gripper type is being used: 0 U2D2, 1 Arduino')

    args = parser.parse_args()
    print(args)
    main(args.gripper_type)
