''' 
WORK IN PROGRESS
The intent of this programme is define a sequence of actions that will rotate a value 90 degrees
using the three fingered grippered being developed and then return the finger to its original position.
This will be used to test the gripper
Beth Cutler 
 '''

import os
import time
import numpy as np
from dynamixel_sdk import *          # Uses Dynamixel SDK library
from gripperFunctions import *

if os.name == 'nt':
    import msvcrt

    def getch():
        return msvcrt.getch().decode()

else:
    import sys
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# region *** Globals ***

# addresses for motor settings
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
ADDR_TORQUE_LIMIT = 35
ADDR_PRESENT_POSITION = 37

PROTOCOL_VERSION = 2.0
BAUDRATE = 57600

NUM_MOTORS = 9

DEVICENAME = 'COM5'

TORQUE_ENABLE = 1  # Value for enabling the torque
TORQUE_DISABLE = 0  # Value for disabling the torque

# setting max velocity and torque
MAX_VELOCITY_VALUE = 80   # Max possible value=2047
LIM_TORQUE_VALUE = 180    # Max possible value=1023

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

targetPos = 180

# endregion

# region *** Setting up the Port ***
# open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

# set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

# endregion

# region *** Making sure torque is enabled, and moving speed and torque are limited ***

limitTorque(NUM_MOTORS, ADDR_TORQUE_LIMIT, LIM_TORQUE_VALUE, packetHandler, portHandler)
enableTorque(NUM_MOTORS, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, packetHandler, portHandler)
limitSpeed(NUM_MOTORS, ADDR_MOVING_SPEED,MAX_VELOCITY_VALUE, packetHandler, portHandler)


# endregion

# region *** Figuring out how to hardcode position, especially if i want to change position value ***
dataLength = 2
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, dataLength)
groupSyncRead = GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POSITION, dataLength)


#set goal values to move to
#these are extremely hard coded and specific to the system, formed from trial and error
#one finger moves, 

jointPos = np.array([[512, 300, 300, 400, 400, 512, 512, 512], #1 base plate
                     [512, 400, 400, 570, 570, 300, 512, 512], #2 middle
                     [512, 400, 400, 370, 230, 230, 512, 512], #3 finger tip

                     [512, 350, 420, 420, 420, 512, 512, 512], #4 baseplate
                     [460, 500, 650, 550, 400, 250, 400, 460], #5 middle
                     [512, 400, 400, 300, 230, 200, 512, 512], #6 finger tip

                     [512, 350, 350, 350, 350, 512, 512, 512], #7 baseplate
                     [512, 400, 400, 400, 512, 512, 512, 512], #8 middle 
                     [512, 400, 400, 400, 512, 512, 512, 512]]) #9 fingertip

#move to goal position
for j in range(0, np.shape(jointPos)[1]): #number of actions
    
    for i in range(0, NUM_MOTORS): #number of motors
        dxl_addparam_result = groupSyncWrite.addParam(i+1, [DXL_LOBYTE(jointPos[i, j]), DXL_HIBYTE(jointPos[i, j])])
        # check for success
        if dxl_addparam_result != True:
            print("all actions complete")
            quit()
     
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result == COMM_SUCCESS:
        print("GroupSyncWrite Succeeded")

    time.sleep(1.5)
    currentPositionToAngle(NUM_MOTORS, ADDR_PRESENT_POSITION, groupSyncRead)
    groupSyncWrite.clearParam()

#endregion

#clear port, disable torque
portHandler.closePort()
#disableTorque(NUM_MOTORS, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, packetHandler, portHandler)
# endregion
