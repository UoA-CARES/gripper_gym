'''
Gripper Class intended to work with the reinforcement learning package being developed
by the University of Auckland Robotics Lab

Beth Cutler

TODO: add all of the action space methods --> to me it makes the most sense to pull them from the servo class 
make it a child class? im very confused how im supposed to this ohhhhhh sample is a python thingy so i just need an action space variable
'''

import numpy as np
import dynamixel_sdk as dxl
from Servo import Servo
from Camera import Camera

class Gripper(object):
    def __init__(self):

        TORQUE_LIMIT = 180  
        SPEED_LIMIT = 100
        DEVICE_NAME = "COM5"  # need to change if in linux, will be dependent on the system

        # is it best practice to do these as class variables? or do I just make them variables that get intialised in the __init__ function?
        self.num_motors = 9
        self.motor_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.motor_positions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.baudrate = 57600
        self.devicename = DEVICE_NAME  # need to change if in linux, will be dependent on the system
        self.protocol = 2.0  # XL-320 uses protocol 2
        self.addresses = {
            "torque_enable": 24,
            "torque_limit": 35,
            "led": 25,
            "goal_position": 30,
            "present_position": 37,
            "present_velocity": 38,
            "moving_speed": 32,
            "moving": 49
        }
        

        self.port_handler = dxl.PortHandler(self.devicename)
        self.packet_handler = dxl.PacketHandler(self.protocol)

        self.group_sync_write = dxl.GroupSyncWrite(
            self.port_handler, self.packet_handler, self.addresses["goal_position"], 2)
        self.group_sync_read = dxl.GroupSyncRead(
            self.port_handler, self.packet_handler, self.addresses["present_position"], 2)

        # create nine servo instances
        self.servo1 = Servo(
            self.port_handler, self.packet_handler, 0, self.addresses, 1, DEVICE_NAME, TORQUE_LIMIT, SPEED_LIMIT)
        self.servo2 = Servo(
            self.port_handler, self.packet_handler, 3, self.addresses, 2, DEVICE_NAME, TORQUE_LIMIT, SPEED_LIMIT)
        self.servo3 = Servo(
            self.port_handler, self.packet_handler, 2, self.addresses, 3, DEVICE_NAME, TORQUE_LIMIT, SPEED_LIMIT)
        self.servo4 = Servo(
            self.port_handler, self.packet_handler, 0, self.addresses, 4, DEVICE_NAME, TORQUE_LIMIT, SPEED_LIMIT)
        self.servo5 = Servo(
            self.port_handler, self.packet_handler, 7, self.addresses, 5, DEVICE_NAME, TORQUE_LIMIT, SPEED_LIMIT)
        self.servo6 = Servo(
            self.port_handler, self.packet_handler, 5, self.addresses, 6, DEVICE_NAME, TORQUE_LIMIT, SPEED_LIMIT)
        self.servo7 = Servo(
            self.port_handler, self.packet_handler, 0, self.addresses, 7, DEVICE_NAME, TORQUE_LIMIT, SPEED_LIMIT)
        self.servo8 = Servo(
            self.port_handler, self.packet_handler, 4, self.addresses, 8, DEVICE_NAME, TORQUE_LIMIT, SPEED_LIMIT)
        self.servo9 = Servo(
            self.port_handler, self.packet_handler, 6, self.addresses, 9, DEVICE_NAME, TORQUE_LIMIT, SPEED_LIMIT)

        # combine all servo instances in a dictionary so I can iterate by motor id
        self.servos = [self.servo1, self.servo2, self.servo3, self.servo4, self.servo5, self.servo6, self.servo7, self.servo8, self.servo9]
        #set up camera instance
        self.camera = Camera()

    def setup(self):

        # open port, set baudrate
        if self.port_handler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()
        # set port baudrate
        if self.port_handler.setBaudRate(self.baudrate):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()

        # setup certain specs of the servos
        for servo in self.servos:
            servo.limit_torque()
            servo.limit_speed()
            servo.enable_torque()
            servo.turn_on_LED()

    """ 
    actions is an array of joint positions that specifiy the desired position of each servo
    this will be a 9 element of array of integers between 0 and 1023
    
    """

    def move(self, action, target_angle):
        #TODO add the reward in here somewhere? 
        
        # loop through the actions array and send the commands to the motors
        if  action.shape[0] != self.num_motors:
           print("Error: action array is not the correct length")
           #quit()

        #print(action)
        for servo in self.servos:
            # add parameters to the groupSyncWrite
            print(action)
            self.group_sync_write.addParam(servo.motor_id, [
                dxl.DXL_LOBYTE(action[servo.motor_id-1]), dxl.DXL_HIBYTE(action[servo.motor_id-1])])

        pre_valve_angle = self.camera.get_marker_pose(0)[0]
        # transmit the packet
        dxl_comm_result = self.group_sync_write.txPacket()
        if dxl_comm_result == dxl.COMM_SUCCESS:
            print("group_sync_write Succeeded")

        self.group_sync_write.clearParam()
        
        post_valve_angle = self.camera.get_marker_pose(0)[0]
        reward = self.reward_function(target_angle, pre_valve_angle, post_valve_angle)
        
        #using moving flag to check if the motors have reached their goal position
        while self.gripper_moving_check():
            self.camera.detect_display()
            self.all_current_positions()
        
        #once its done, append the valve position to the list and return? at least return current position
        return self.all_current_positions(), reward


    
    def reward_function(self, target_angle, valve_angle_previous, valve_angle_after):
            delta_changes    = np.abs(target_angle - valve_angle_previous) - np.abs(target_angle - valve_angle_after)
            angle_difference = np.abs(target_angle - valve_angle_after)

            if -5 <= delta_changes <= 5:
                # noise or no changes
                reward_ext = 0
            else:
                reward_ext = delta_changes

            if angle_difference <= 5:
                
                reward_ext = reward_ext + 100
                print(f"reward: {reward_ext}")
                done = True
            else:
                done = False

            return reward_ext, done, angle_difference



    def all_current_positions(self):

        #set up a txrx packet 
        self.group_sync_read.txRxPacket()
        
        #add parameters to the groupSyncRead
        for servo in self.servos:
            self.group_sync_read.addParam(servo.motor_id)

        #get the data from the groupSyncRead
        for servo in self.servos:
            currentPos = self.group_sync_read.getData(
                servo.motor_id, self.addresses["present_position"], 2)
            self.motor_positions[servo.motor_id -
                                 1] = np.round(((0.2929 * currentPos) - 150), 1)
        #print(self.motor_positions)

        all_current_positions = np.append(self.motor_positions, self.camera.get_marker_pose(0)[0])

        return all_current_positions


    def reset(self):
        
        print("got to reset")
        reset_seq = np.array([[512, 512],  # 1 base plate
                              [185, 512],  # 2 middle
                              [126, 512],  # 3 finger tip

                              [512, 512],  # 4 baseplate
                              [143, 460],  # 5 middle
                              [150, 512],  # 6 finger tip

                              [512, 512],  # 7 baseplate
                              [188, 512],  # 8 middle
                              [138, 512]]) # 9 finger tip
        self.move(reset_seq[:,0],0)
        self.move(reset_seq[:,1],0)

        #TODO: append valve position to all current positions then return as state
        state = self.all_current_positions()

        #can't append angle because then each time the thing resets it will add it on. 
        #theres gotta be a better way to do the observation space stuff
        #maybe i will include it in the get_all_current_positions method. ye   
        
        return state

        

    def gripper_moving_check(self):
        moving = False
        for servo in self.servos:
            moving |= servo.moving_check()
        return moving

    def close(self):

        # disable torque
        for servo in self.servos:
            servo.disable_torque()
        # close port
        self.port_handler.closePort()

    
