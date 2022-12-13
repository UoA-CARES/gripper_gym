'''
Gripper Class intended to work with the summer reinforcement learning package being developed
by the University of Auckland Robotics Lab

Author: Beth Cutler
'''
import numpy as np
import gripperFunctions as gf
import dynamixel_sdk as dxl
import time

class Gripper9():
    def __init__(self):

        #is it best practice to do these as class variables? or do I just make them variables that get intialised in the __init__ function?
        self.num_motors = 9
        self.motor_ids = [1,2,3,4,5,6,7,8,9]
        self.motor_positions = [0,0,0,0,0,0,0,0,0]
        self.baudrate = 57600
        self.devicename = "COM5"
        self.protocol = 2.0
        self.addresses = {
            "torque_enable": 24,
            "torque_limit": 35,
            "led": 25,
            "goal_position": 30,
            "present_position": 37,
            "present_velocity": 38,
            "moving_speed": 32
        }
        self.torque_limit = 180

        self.portHandler = dxl.PortHandler(self.devicename)
        self.packetHandler = dxl.PacketHandler(self.protocol)

        #figure out how I want to do groupSyncRead and groupSyncWrite, because they'll 


    def setup(self):

        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()
        # set port baudrate
        if self.portHandler.setBaudRate(self.baudrate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()

        gf.limitTorque(self.num_motors, self.addresses["torque_limit"], self.torque_limit, self.packetHandler, self.portHandler)
        gf.limitSpeed(self.num_motors, self.addresses["moving_speed"], 80, self.packetHandler, self.portHandler)
        gf.enableTorque(self.num_motors, self.addresses["torque_enable"], 1, self.packetHandler, self.portHandler)
        gf.turnOnLEDS(self.num_motors, self.addresses["led"], self.packetHandler, self.portHandler)  #current only works for 9 motors and how ive specifically got it setup


    def move(self, actions):

        #TODO potientially add a check in here that clips the actions given from the network just to set some kind of limit
    
        #set up read write groups
        groupSyncWrite = dxl.GroupSyncWrite(
        self.portHandler, self.packetHandler, self.addresses["goal_position"], 2)
        groupSyncRead = dxl.GroupSyncRead(
        self.portHandler, self.packetHandler, self.addresses["present_position"], 2) 
        #move the servos to the desired positions
        for j in range(0, np.shape(actions)[1]):  # number of actions
            for i in range(0, self.num_motors):  # number of motors
                dxl_addparam_result = groupSyncWrite.addParam(
                    i+1, [dxl.DXL_LOBYTE(actions[i, j]), dxl.DXL_HIBYTE(actions[i, j])])

            dxl_comm_result = groupSyncWrite.txPacket()
            if dxl_comm_result == dxl.COMM_SUCCESS:
                print("GroupSyncWrite Succeeded")

            time.sleep(1.5)
            gf.currentPositionToAngle(self.num_motors, self.addresses["present_position"], groupSyncRead)
            groupSyncWrite.clearParam()

    def current_positions(self):

        groupSyncRead = dxl.GroupSyncRead(
        self.portHandler, self.packetHandler, self.addresses["present_position"], 2)   
        self.motor_positions = gf.currentPositionToAngle(self.num_motors, self.addresses["present_position"], groupSyncRead)

        return self.motor_positions
    