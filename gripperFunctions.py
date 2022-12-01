import numpy as np
from dynamixel_sdk import *

"""
The following are commonly used and repeattive functions that are used in the gripper test code.
These have been collated to make the code more readable and easier to maintain.
Beth Cutler
"""


# limit the torque of the motors
def limitTorque(NUM_MOTORS, ADDR_TORQUE_LIMIT, LIM_TORQUE_VALUE, packetHandler, portHandler):
    for i in range(1, NUM_MOTORS+1):
        # write and read to servos
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
            portHandler, i, ADDR_TORQUE_LIMIT, LIM_TORQUE_VALUE)
        # verify write read successful
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully torque limited" % i)

# enable the torque of the motors


def enableTorque(NUM_MOTORS, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, packetHandler, portHandler):
    for i in range(1, NUM_MOTORS+1):
        # write and read to servos
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        # verify write read successful
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully torque enabled" % i)

# disable the torque of the motors


def disableTorque(NUM_MOTORS, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, packetHandler, portHandler):
    for i in range(1, NUM_MOTORS+1):
        # write and read to servos
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler, i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        # verify write read successful
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully torque disabled" % i)

# limit the speed of the servos


def limitSpeed(NUM_MOTORS, ADDR_MOVING_SPEED, MAX_VELOCITY_VALUE, packetHandler, portHandler):
    for i in range(1, NUM_MOTORS+1):
        # write and read to servos
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
            portHandler, i, ADDR_MOVING_SPEED, MAX_VELOCITY_VALUE)
        # verify write read successful
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully speed limited" % i)


# read the current position from all servos and return them in a list
# angles go from -150 deg to 150 deg, where 512 corresponds to 0 deg
def currentPositionToAngle(NUM_MOTORS, ADDR_PRESENT_POSITION, groupSyncRead):

    currentPos = np.zeros(NUM_MOTORS)
    # get current position of all currently operating motors
    groupSyncRead.txRxPacket()

    for i in range(1, NUM_MOTORS+1):

        dxl_addparam_result = groupSyncRead.addParam(i)
        # read data, covert and
        currentPos[i-1] = groupSyncRead.getData(i, ADDR_PRESENT_POSITION, 2)
        currentPos[i-1] = np.round(((0.2929*currentPos[i-1])-150), 1)
    return print(currentPos)
