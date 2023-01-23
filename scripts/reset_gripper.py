import logging
logging.basicConfig(level=logging.INFO)

import time
import numpy as np
from Gripper import Gripper
from cares_lib.dynamixel.Servo import DynamixelServoError

def main():
        
    gripper = Gripper()

    try:
        state = gripper.home()
    except DynamixelServoError as error:
        print("Gripper Failed to home")
        return
    
    poses = []
    poses.append([550, 300, 700, 550, 300, 700, 550, 300, 700])
    poses.append([512, 250, 750, 512, 250, 750, 512, 250, 750])

    pose = 0
    for i in range(0, 50):
        try:
            gripper.move(steps=poses[pose])
        except DynamixelServoError:
            print("Gripper Failed to move - ressetting")
        pose = i % 2

    # gripper.move_servo(0, 600, wait=True)
    # gripper.move_servo(3, 600, wait=True)
    # gripper.move_servo(6, 600, wait=True)



    logging.info(f"State: {gripper.current_positions()}")

if __name__ == '__main__':
    main()