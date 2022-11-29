#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Description:
            For testing only and see if the robot can rotate the object
            Here nothing RL or learning. all is a sequence of movements
            Modified from original file read_write from Dynamixel SDK-Python library
                Motor: MY_DXL ---> 'XL320'
                Bps = 1 Mbps
                Min steps  = 0
                Max steps  = 1023
                Max Degree = 300
                Resolution [deg/pulse] = 0.2930
                Move 4 servos at the same time
                Additionally speed and torque are limited
"""

import os
from dynamixel_sdk import *  # Uses Dynamixel SDK library

if os.name == 'nt':
    import msvcrt

    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch



ADDR_TORQUE_ENABLE      = 24
ADDR_GOAL_POSITION      = 30
ADDR_MOVING_SPEED       = 32
ADDR_TORQUE_LIMIT       = 35
ADDR_PRESENT_POSITION   = 37


# Protocol and Bps
PROTOCOL_VERSION = 2.0
BAUDRATE         = 57600  # Default Baudrate of XL-320 is 1Mbps

DXL_ID_1 = 1
DXL_ID_2 = 2
DXL_ID_3 = 3
DXL_ID_4 = 4
DXL_ID_5 = 5
DXL_ID_6 = 6
DXL_ID_7 = 7
DXL_ID_8 = 8
DXL_ID_9 = 9

# Use the actual port assigned to the U2D2.
DEVICENAME = 'COM5'

TORQUE_ENABLE  = 1  # Value for enabling the torque
TORQUE_DISABLE = 0  # Value for disabling the torque

# Speed values
DXL_MAX_VELOCITY_VALUE = 100   # Max possible value=2047
DXL_LIM_TORQUE_VALUE   = 180    # Max possible value=1023


# GOAL VALUES FOR EACH MOTOR
index = 0
id_1_dxl_goal_position = [500,500]  # Goal position for motor 1 (min max)
id_2_dxl_goal_position = [500, 550]  # Goal position for motor 2
id_3_dxl_goal_position = [500, 400]  # Goal position for motor 3
id_4_dxl_goal_position = [500, 500]  # Goal position for motor 2
id_5_dxl_goal_position = [500, 550]  # Goal position for motor 3
id_6_dxl_goal_position = [500, 450]  # Goal position for motor 2
id_7_dxl_goal_position = [500, 500]  # Goal position for motor 3
id_8_dxl_goal_position = [500, 550]  # Goal position for motor 2
id_9_dxl_goal_position = [500, 450]  # Goal position for motor 3


portHandler   = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# ------------------------Open port-------------------------
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()
# ----------------------------------------------------------

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

#region ---------------Enable Torque Limit for each motor-----------------------------------------------------------
# This should be here because if torque is enable=1 we can not change the max torque again

print("enabling torque limits")

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_1, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_1)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_2, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_2)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_3, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_3)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_4, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_4)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_5, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_5)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_6, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_6)    

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_7, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_7)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_8, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_8)    

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_9, ADDR_TORQUE_LIMIT, DXL_LIM_TORQUE_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the TORQUE" % DXL_ID_9)

#endregion

#region ---------------Enable Torque for each motor-----------------------------------------------------------

print("enabling torque")

# Enable Dynamixel#1 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_1, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_1)

# Enable Dynamixel#2 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_2, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_2)

# Enable Dynamixel#3 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_3, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_3)

# Enable Dynamixel#4 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_4, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_4)

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_5, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_5)

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_6, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_6)

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_7, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_7)    

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_8, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_8)

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_9, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully connected" % DXL_ID_9)

#endregion

#region ---------------Enable Moving Speed for each motor-----------------------------------------------------------

print("setting moving speed")

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_1, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_1)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_2, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_2)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_3, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_3)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_4, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_4)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_5, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_5)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_6, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_6)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_7, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_7)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_8, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_8)

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID_9, ADDR_MOVING_SPEED,
                                                          DXL_MAX_VELOCITY_VALUE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully limited the speed" % DXL_ID_9)

# endregion ------------------------------------------------------------------------------------------------------------


# ---------------------Initialize GroupSyncWrite instance --------------------

# Initialize GroupSyncWrite instance ---> GroupSyncWrite(port, ph, start_address, data_length)

data_length = 2  # data length goal position  and present position

groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, data_length)

# syncwrite test start

while 1:

    print(index)
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1b):
        break

    # The Size of the goal position is 2 bytes

    param_goal_position_1 = [DXL_LOBYTE(id_1_dxl_goal_position[index]), DXL_HIBYTE(id_1_dxl_goal_position[index])]
    param_goal_position_2 = [DXL_LOBYTE(id_2_dxl_goal_position[index]), DXL_HIBYTE(id_2_dxl_goal_position[index])]
    param_goal_position_3 = [DXL_LOBYTE(id_3_dxl_goal_position[index]), DXL_HIBYTE(id_3_dxl_goal_position[index])]
    param_goal_position_4 = [DXL_LOBYTE(id_4_dxl_goal_position[index]), DXL_HIBYTE(id_4_dxl_goal_position[index])]
    param_goal_position_5 = [DXL_LOBYTE(id_5_dxl_goal_position[index]), DXL_HIBYTE(id_5_dxl_goal_position[index])]
    param_goal_position_6 = [DXL_LOBYTE(id_6_dxl_goal_position[index]), DXL_HIBYTE(id_6_dxl_goal_position[index])]
    param_goal_position_7 = [DXL_LOBYTE(id_7_dxl_goal_position[index]), DXL_HIBYTE(id_7_dxl_goal_position[index])]
    param_goal_position_8 = [DXL_LOBYTE(id_8_dxl_goal_position[index]), DXL_HIBYTE(id_8_dxl_goal_position[index])]
    param_goal_position_9 = [DXL_LOBYTE(id_9_dxl_goal_position[index]), DXL_HIBYTE(id_9_dxl_goal_position[index])]

    #region----adding goal values to the syncwrite group--------------------------


    # --- Add the goal position value to the Syn parameter, motor ID1 ----
    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_1, param_goal_position_1)

    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_1)
        quit()

    # --- Add the goal position value to the Syn parameter, motor ID2 ----
    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_2, param_goal_position_2)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_2)
        quit()

    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_3, param_goal_position_3)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_3)
        quit()

    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_4, param_goal_position_4)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_4)
        quit()

    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_5, param_goal_position_5)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_5)
        quit()

    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_6, param_goal_position_6)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_6)
        quit()

    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_7, param_goal_position_7)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_7)
        quit()

    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_8, param_goal_position_8)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_8)
        quit()

    dxl_addparam_result = groupSyncWrite.addParam(DXL_ID_9, param_goal_position_9)
    if dxl_addparam_result != True:
        print("[ID:%03d] groupSyncWrite addparam failed" % DXL_ID_9)
        quit()


    #endregion-----------------------------------------------------------------

    # ---- Transmits packet (goal position) to the motors
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))

    # Clear syncwrite parameter storage
    groupSyncWrite.clearParam()

    # Change goal position index
    if index == 0:
        index = 1  
    else:
        index = 0


# Disable communication and close the port

#region ---------------Disable Torque for each motor-----------------------
# Enable Dynamixel_1 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_1, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_1)

# Enable Dynamixel_2 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_2, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_2)

# Enable Dynamixel_3 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_3, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_3)

# Enable Dynamixel_4 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_4, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_4)

# Enable Dynamixel_5 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_5, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_5)

# Enable Dynamixel_6 Torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_6, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)    
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_6)

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_7, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_7)

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_8, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_8)

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_9, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel#%d has been successfully disable torque" % DXL_ID_9)
#endregion

# Close port
portHandler.closePort()
print("Succeeded to close the USB port ")