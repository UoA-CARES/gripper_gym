'''
Gripper Class intended to work with the summer reinforcement learning package being developed
by the University of Auckland Robotics Lab

Author: Beth Cutler
'''
import numpy as np
import gripperFunctions as gf
import dynamixel_sdk as dxl
import time
from gripperFunctions import Servo

class Gripper9(object):
    def __init__(self):

        #is it best practice to do these as class variables? or do I just make them variables that get intialised in the __init__ function?
        self.num_motors = 9
        self.motor_ids = [1,2,3,4,5,6,7,8,9]
        self.motor_positions = [0,0,0,0,0,0,0,0,0]
        self.baudrate = 57600
        self.devicename = "COM5"   #need to change if ur in linux
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

        self.groupSyncWrite = dxl.GroupSyncWrite(
            self.portHandler, self.packetHandler, self.addresses["goal_position"], 2)
        self.groupSyncRead = dxl.GroupSyncRead(
            self.portHandler, self.packetHandler, self.addresses["present_position"], 2) 
        #figure out how I want to do groupSyncRead and groupSyncWrite, because they may fuck up

        #create nine servo instances 
        self.servo1 = Servo(self.portHandler, self.packetHandler, 0, self.addresses, 1)
        self.servo2 = Servo(self.portHandler, self.packetHandler, 3, self.addresses, 2)
        self.servo3 = Servo(self.portHandler, self.packetHandler, 2, self.addresses, 3)
        self.servo4 = Servo(self.portHandler, self.packetHandler, 0, self.addresses, 4)
        self.servo5 = Servo(self.portHandler, self.packetHandler, 7, self.addresses, 5)
        self.servo6 = Servo(self.portHandler, self.packetHandler, 5, self.addresses, 6)
        self.servo7 = Servo(self.portHandler, self.packetHandler, 0, self.addresses, 7)
        self.servo8 = Servo(self.portHandler, self.packetHandler, 4, self.addresses, 8)
        self.servo9 = Servo(self.portHandler, self.packetHandler, 6, self.addresses, 9)

        self.servos = [self.servo1, self.servo2, self.servo3, self.servo4, self.servo5, self.servo6, self.servo7, self.servo8, self.servo9]

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

        for servo in self.servos:
            servo.limit_torque()
            servo.limit_speed()
            servo.enable_torque()
            servo.turn_on_LED()

    def move(self, actions):

        #TODO potientially add a check in here that clips the actions given from the network just to set some kind of limit
    
        #set up read write groups
        
        #move the servos to the desired positions
        for j in range(0, np.shape(actions)[1]):  # number of actions
            for i in range(0, self.num_motors):  # number of motors
                self.groupSyncWrite.addParam(i+1, [dxl.DXL_LOBYTE(actions[i, j]), dxl.DXL_HIBYTE(actions[i, j])])

            dxl_comm_result = self.groupSyncWrite.txPacket()
            if dxl_comm_result == dxl.COMM_SUCCESS:
                print("GroupSyncWrite Succeeded")

            time.sleep(1.5)
    
            self.groupSyncWrite.clearParam()

    def current_positions(self):
  
        self.groupSyncRead.txRxPacket()
        #better way to do this?
        for servo in self.servos:
            self.groupSyncRead.addParam(servo.motor_id)
        for servo in self.servos:
            currentPos = self.groupSyncRead.getData(servo.motor_id, self.addresses["present_position"], 2)
            self.motor_positions[servo.motor_id - 1] = np.round(((0.2929 * currentPos) - 150), 1)
        print(self.motor_positons)
        
        return self.motor_positions
    