''' 
WORK IN PROGRESS
The intent of this programme is define a sequence of actions that will rotate a value 90 degrees
using the three fingered grippered being developed and then return the finger to its original position.
This will be used to test the gripper

--> #TODO make gripper class (later, basically make a copy but join everything together)

Beth Cutler 
 '''

import time
import numpy as np
import dynamixel_sdk as dxl         # Uses Dynamixel SDK library
import gripperFunctions as gf
from GripperClass import Gripper9


# region *** Globals ***

# addresses for motor settings
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
ADDR_TORQUE_LIMIT = 35
ADDR_PRESENT_POSITION = 37
ADDR_LED = 25

PROTOCOL_VERSION = 2.0
BAUDRATE = 57600

NUM_MOTORS = 9

DEVICENAME = 'COM5'

TORQUE_ENABLE = 1  # Value for enabling the torque
TORQUE_DISABLE = 0  # Value for disabling the torque

# setting max velocity and torque
MAX_VELOCITY_VALUE = 80   # Max possible value=2047
LIM_TORQUE_VALUE = 180    # Max possible value=1023


portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

jointPos = np.array([[512, 300, 300, 400, 400, 512, 512],  # 1 base plate
                        [512, 400, 400, 570, 570, 300, 512],  # 2 middle
                        [512, 400, 400, 370, 230, 200, 512],  # 3 finger tip

                        [512, 350, 420, 420, 420, 512, 512],  # 4 baseplate
                        [460, 500, 650, 550, 400, 250, 460],  # 5 middle
                        [512, 400, 400, 300, 230, 200, 512],  # 6 finger tip

                        [512, 350, 350, 350, 350, 512, 512],  # 7 baseplate
                        [512, 400, 400, 400, 512, 512, 512],  # 8 middle
                        [512, 400, 400, 400, 512, 512, 512]])  # 9 fingertip

# endregion

def main():


    gripper = Gripper9()
    gripper.setup()
    #while True:
    gripper.move(jointPos) 

#clear port, disable torque
    gripper.portHandler.closePort()

if __name__ == "__main__":
    main()
