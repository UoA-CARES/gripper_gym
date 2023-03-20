import logging
# logging.basicConfig(level=logging.INFO)

import time
from datetime import datetime
import numpy as np
from ArduinoGripper import Gripper
from cares_lib.dynamixel.Servo import DynamixelServoError

def main():
    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")    
    log_path = f"logs/{now}.log"
    logging.basicConfig(filename=log_path, level=logging.DEBUG)

    gripper = Gripper()
    state = gripper.home()

    gripper.move_servo(0, 600, wait=False)
    gripper.move_servo(3, 600, wait=False)
    gripper.move_servo(6, 600, wait=True)

    poses = []
    poses.append([550, 300, 700, 550, 300, 700, 550, 300, 700])
    poses.append([512, 250, 750, 512, 250, 750, 512, 250, 750])

    pose = 0
    for i in range(0, 50):
        try:
            gripper.move(steps=poses[pose])
        except DynamixelServoError:
            print("Gripper Failed to move - now we have to handle what went wrong...")
        pose = i % 2

    logging.info(f"State: {gripper.current_positions()}")

if __name__ == '__main__':
    main()