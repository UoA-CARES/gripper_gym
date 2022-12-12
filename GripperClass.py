'''
Gripper Class intended to work with the summer reinforcement learning package being developed
by the University of Auckland Robotics Lab

Author: Beth Cutler
'''
import numpy as np
import gripperFunctions as gf
import dynamixel_sdk as dxl

class gripper9():
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

    def move(self, actions):

        #TODO potientially add a check in here that clips the actions given from the network just to set some kind of limit
        
        #setup the servos
        gf.setup(self.baudrate, self.protocol, self.devicename)
        gf.limitTorque(self.num_motors, self.addresses["torque_limit"], self.torque_limit, self.protocol, self.devicename)
        gf.enableTorque(self.num_motors, self.addresses["torque_enable"], 1, self.protocol, self.devicename)
        gf.turnOnLEDS(self.num_motors, self.addresses["led"], self.protocol, self.devicename)  #current only works for 9 motors and how ive specifically got it setup

        #set up read write groups
        groupSyncWrite = dxl.GroupSyncWrite(
        self.portHandler, self.packetHandler, self.addresses["present_position"], 2)
        groupSyncRead = dxl.GroupSyncRead(
        self.portHandler, self.packetHandler, self.addresses["present_position"], 2) 

        #move the servos to the desired positions
        for j in range(0, np.shape(actions)[1]): 
            for i in range(0, self.num_motors):
                groupSyncWrite.addParam(self.motor_ids[i], [dxl.DXL_LOBYTE(actions[i,j]), dxl.DXL_HIBYTE(actions[i,j])])
            
            #write to servos
            groupSyncWrite.txPacket()
            #print the current position as an angle, and to verify the success of the write
            gf.currentPositionToAngle(self.num_motors, self.addresses["present_position"], groupSyncRead)
            groupSyncWrite.clearParam()           


    def current_positions(self):

        groupSyncRead = dxl.GroupSyncRead(
        self.portHandler, self.packetHandler, self.addresses["present_position"], 2)   
        self.motor_positions = gf.currentPositionToAngle(self.num_motors, self.addresses["present_position"], groupSyncRead)
        
        return self.motor_positions
    