'''
Gripper Class intended to work with the reinforcement learning package being developed
by the University of Auckland Robotics Lab

Beth Cutler

'''
import logging

import time
import numpy as np
import dynamixel_sdk as dxl
from Camera import Camera
from cares_lib.dynamixel.Servo import Servo, DynamixelServoError

class Gripper(object):
    def __init__(self, device_name="/dev/ttyUSB0", baudrate=115200, protocol=2.0, torque_limit=180, speed_limit=100):
        # Setup Servor handlers
        self.device_name = device_name
        self.baudrate = baudrate
        self.protocol = protocol  # NOTE: XL-320 uses protocol 2

        self.port_handler = dxl.PortHandler(self.device_name)
        self.packet_handler = dxl.PacketHandler(self.protocol)
        try:
            self.setup_handlers()
        except IOError as error:
            raise IOError("Failed to initialise port and packet handler") from error

        self.group_sync_write = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, Servo.addresses["goal_position"], 2)
        self.group_sync_read  = dxl.GroupSyncRead(self.port_handler, self.packet_handler,  Servo.addresses["current_position"], 2)
        
        self.servos = {}
        self.num_motors = 9
        
        leds = [0, 3, 2, 0, 7, 5, 0, 4, 6]# TODO set colour IDs in Servo
        # Ideally these would all be the same but some are slightly physically offset
        # TODO: paramatise this further for when we have multiple grippers
        max = [900, 750, 769, 900, 750, 802, 900, 750, 794]
        min = [100, 250, 130, 100, 198, 152, 100, 250, 140]

        try:
            for i in range(0, self.num_motors):
                self.servos[i] = Servo(self.port_handler, self.packet_handler, leds[i], i+1, torque_limit, speed_limit, max[i], min[i])
                self.setup_servos()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Failed to initialise Gripper servos") from error

    def setup_handlers(self):
        if self.port_handler.openPort():
            logging.info(f"Succeeded to open the port {self.device_name}")
        else:
            error_message = f"Failed to open the port {self.device_name}"
            logging.error(error_message)
            raise IOError(error_message)

        if self.port_handler.setBaudRate(self.baudrate):
            logging.info(f"Succeeded to change the baudrate to {self.baudrate}")
        else:
            error_message = f"Failed to change the baudrate to {self.baudrate}"
            logging.error(error_message)
            raise IOError(error_message)
    
    def setup_servos(self):
        try:
            for _, servo in self.servos.items():
                servo.limit_torque()
                servo.limit_speed()
                servo.enable_torque()
                servo.turn_on_LED()
        except DynamixelServoError as error:
                raise DynamixelServoError(f"Gripper failed to setup servos") from error

    def is_overloaded(self):
        for _, servo in self.servos.items():
            try:
                current_load = servo.current_load() 
                if current_load > 20:#% of maximum load
                    return True
            except DynamixelServoError as error:
                raise DynamixelServoError(f"Gripper failed to check load") from error
        return False

    def stop_moving(self):
        for _, servo in self.servos.items():
            try:
                servo.stop_moving()
            except DynamixelServoError as error:
                raise DynamixelServoError(f"Gripper failed to stop moving") from error

    def is_moving(self):
        gripper_moving = False
        for _, servo in self.servos.items():
            try:
                gripper_moving |= servo.is_moving()
            except DynamixelServoError as error:
                raise DynamixelServoError(f"Gripper failed to check if moving") from error
        return gripper_moving

    def move_servo(self, servo_id, target_step=None, target_angle=None, wait=True):
        if servo_id not in self.servos:
            error_message = f"Dynamixel#{servo_id} is not associated to this gripper"
            logging.error(error_message)
            raise DynamixelServoError(error_message)
        
        if target_angle is not None:
            target_step = Servo.angle_to_step(target_angle)

        if target_step is None:
            error_message = f"No move command given to Dynamixel#{servo_id}: Step {target_step} Angles {target_angle}"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        try:
            servo_pose = self.servos[servo_id].move(target_step, wait=wait)
            return self.current_positions()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper failed while moving Dynamixel#{servo_id}") from error

    def move(self, steps=None, angles=None, action=None, wait=True):
        if angles is not None:
            steps = self.angles_to_steps(angles)

        if action is not None:
            steps = self.action_to_steps(action)

        if steps is None:
            error_message = f"No move command given to Gripper: Step {steps} Angles {angles} Action {action}"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        if not self.verify_steps(steps):
            error_message = f"The move command provided to Gripper is out of bounds: Step {steps} Angles {angles} Action {action}"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        for id, servo  in self.servos.items():
            servo.target_position = steps[id]
            self.group_sync_write.addParam(id+1, [dxl.DXL_LOBYTE(steps[id]), dxl.DXL_HIBYTE(steps[id])])

        dxl_comm_result = self.group_sync_write.txPacket()
        if dxl_comm_result != dxl.COMM_SUCCESS:
            error_message = "group_sync_write Failed"
            logging.error(error_message)
            raise DynamixelServoError(error_message)
        
        logging.debug("group_sync_write Succeeded")
        self.group_sync_write.clearParam()

        # using moving flag to check if the motors have reached their goal position
        try:
            while self.is_moving() and wait:
                if self.is_overloaded():
                    logging.warn("Gripper load too high on one of the servos resetting gripper")
                    self.close()
                    self.setup_servos()
                    break
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper failed while moving") from error
               
        # return the current state and whether it was terminated
        try:
            return self.current_positions()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper failed to read its positions") from error

    def current_positions(self):
        current_positions = []
        for id, servo in self.servos.items():
            servo_position = servo.current_position()
            current_positions.append(servo_position)
        return current_positions

    def home(self):
        home_pose = [512,  # 1 base plate 512
                    250,  # 2 middle
                    750,  # 3 finger tip

                    512,  # 4 baseplate
                    250,  # 5 middle
                    750,  # 6 finger tip

                    512,  # 7 baseplate
                    250,  # 8 middle
                    750]  # 9 finger tip

        try:
            return self.move(steps=home_pose)
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper failed to Home") from error

    def verify_steps(self, steps):
        # check all actions are within min max of each servo
        for id, servo in self.servos.items():
            if not servo.verify_step(steps[id]):
                logging.warn(f"Step for servo {id+1} is out of bounds")
                return False
        return True

    def action_to_steps(self, action):
        steps = action
        for i in range(0, len(steps)):
            max = self.servos[i].max
            min = self.servos[i].min
            steps[i] = steps[i] * (max - min) + min
        return steps

    def steps_to_angles(self, steps):
        angles = []
        for i in range(0,len(steps)):
            angles.append(Servo.step_to_angle(steps[i]))
        return angles

    def angles_to_steps(self, angles):
        steps = []
        for i in range(0,len(steps)):
            steps.append(Servo.angle_to_step(steps[i]))
        return steps

    def close(self):
        # disable torque
        for _, servo in self.servos.items():
            servo.disable_torque()
        # close port
        self.port_handler.closePort()