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

# endregion

def main():
# region *** Setting up the Port ***
# open port
    gf.setup(BAUDRATE, portHandler)
    # endregion

    # region *** Making sure torque is enabled, and moving speed and torque are limited **# *
    gf.enableTorque(NUM_MOTORS, ADDR_TORQUE_ENABLE,
             TORQUE_ENABLE, packetHandler, portHandler)
    gf.limitTorque(NUM_MOTORS, ADDR_TORQUE_LIMIT,
            LIM_TORQUE_VALUE, packetHandler, portHandler)
    
    gf.limitSpeed(NUM_MOTORS, ADDR_MOVING_SPEED,
           MAX_VELOCITY_VALUE, packetHandler, portHandler)
    # endregion

    #turn on the leds 
    gf.turnOnLEDS(NUM_MOTORS, ADDR_LED, packetHandler, portHandler)

    # region *** Figuring out how to hardcode position, especially if i want to change position value ***
    dataLength = 2
    groupSyncWrite = dxl.GroupSyncWrite(
        portHandler, packetHandler, ADDR_GOAL_POSITION, dataLength)
    groupSyncRead = dxl.GroupSyncRead(
        portHandler, packetHandler, ADDR_PRESENT_POSITION, dataLength)


    # set goal values to move to
    # these are extremely hard coded and specific to the system, formed from trial and error
    # one finger moves,

    jointPos = np.array([[512, 300, 300, 400, 400, 512, 512],  # 1 base plate
                        [512, 400, 400, 570, 570, 300, 512],  # 2 middle
                        [512, 400, 400, 370, 230, 200, 512],  # 3 finger tip

                        [512, 350, 420, 420, 420, 512, 512],  # 4 baseplate
                        [460, 500, 650, 550, 400, 250, 400],  # 5 middle
                        [512, 400, 400, 300, 230, 200, 512],  # 6 finger tip

                        [512, 350, 350, 350, 350, 512, 512],  # 7 baseplate
                        [512, 400, 400, 400, 512, 512, 512],  # 8 middle
                        [512, 400, 400, 400, 512, 512, 512]])  # 9 fingertip

    # move to goal position

    while True: #continuous testing
        for j in range(0, np.shape(jointPos)[1]):  # number of actions

            for i in range(0, NUM_MOTORS):  # number of motors
                dxl_addparam_result = groupSyncWrite.addParam(
                    i+1, [dxl.DXL_LOBYTE(jointPos[i, j]), dxl.DXL_HIBYTE(jointPos[i, j])])

            dxl_comm_result = groupSyncWrite.txPacket()
            if dxl_comm_result == dxl.COMM_SUCCESS:
                print("GroupSyncWrite Succeeded")

            time.sleep(1.5)
            gf.currentPositionToAngle(NUM_MOTORS, ADDR_PRESENT_POSITION, groupSyncRead)
            groupSyncWrite.clearParam()

    # endregion

    # clear port, disable torque
    portHandler.closePort()

if __name__ == "__main__":
    main()
