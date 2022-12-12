'''
Gripper Class intended to work with the summer reinforcement learning package being developed
by the University of Auckland Robotics Lab

Author: Beth Cutler
'''
import numpy as np
import gripperFunctions as gF

class gripper9():
    def __init__(self):
        self.num_motors = 9
        self.motor_ids = [1,2,3,4,5,6,7,8,9]
        self.motor_positions = [0,0,0,0,0,0,0,0,0]
        self.baudrate = 57600
        self.devicename = "COM5"
        self.protocol = 2.0

    def move(self, actions):
        
        #setup
        gF.setup(self.baudrate, self.protocol, self.devicename)
        gF.limitTorque(self.num_motors, gF.ADDR_TORQUE_LIMIT, gF.LIM_TORQUE_VALUE, gF.packetHandler, gF.portHandler)
        gF.enableTorque(self.num_motors, gF.ADDR_TORQUE_ENABLE, gF.TORQUE_ENABLE, gF.packetHandler, gF.portHandler)
        gF.turnOnLEDS(self.num_motors, gF.ADDR_LED, gF.packetHandler, gF.portHandler)

    def current_positions(self):

        return #all the joint positions (in degrees or steps?)
    