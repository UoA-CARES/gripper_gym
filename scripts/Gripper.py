'''
Gripper Class intended to work with the reinforcement learning package being developed
by the University of Auckland Robotics Lab

Beth Cutler

'''
import logging

import serial
import time
import numpy as np
# from Camera import Camera

from enum import Enum, auto

class Command(Enum):
    STOP = 0
    MOVE = 1
    GET_STATE = 2

class Response(Enum):
    SUCCEEDED = 0
    ERROR_STATE = 1
    TIMEOUT = 2

class GripperError(IOError):
    pass

class Gripper(object):
    def __init__(self, gripper_id=0, device_name="/dev/ttyACM0", baudrate=115200):
        # Setup Servor handlers
        self.gripper_id = gripper_id
        self.device_name = device_name
        self.baudrate = baudrate
        self.arduino = serial.Serial(device_name, baudrate)

    def process_response(self, response):
      if '\n' not in response:
        logging.debug(f"Serial Read Timeout")
        return Response.TIMEOUT
      error_state = int(response.split(',')[0])
      logging.debug(f"Error Flag: {error_state} {Response(error_state)}")
      return Response(error_state)

    def current_positions(self,timeout=5):
        command = f"{Command.GET_STATE.value}"
        logging.debug(f"Command: {command}")

        try:
            self.arduino.write(command.encode())
        except serial.SerialException as error:
            raise GripperError(f"Gripper#{self.gripper_id} failed to write to arduino {self.device_name}") from error

        self.arduino.timeout = timeout
        response = self.arduino.read_until(b'\n').decode()
        logging.debug(f"Response: {response}")
        
        comm_result = self.process_response(response)
        if comm_result != Response.SUCCEEDED:
            raise GripperError(f"Gripper#{self.gripper_id}: {comm_result} getting current position")

        state = [int(x) for x in response.split(',')[1:]] 
        return state

    def stop_moving(self,timeout=5):
        try:
            command = f"{Command.STOP.value}"
            self.arduino.write(command.encode())
            logging.debug(f"Command: {command}")

            self.arduino.timeout = timeout
            response = self.arduino.read_until(b'\n').decode()
            comm_result = self.process_response(response)
            logging.debug(f"Response: {response}")

            if comm_result != Response.SUCCEEDED:
                raise GripperError(f"Gripper#{self.gripper_id}: {comm_result} while stopping")

            state = [int(x) for x in response.split(',')[1:]] 
            return state              
            
        except GripperError as error:
            raise GripperError(f"Gripper#{self.gripper_id}: failed to stop moving") from error

    def move_servo(self, servo_id, target_step, timeout=5):
        raise NotImplementedError("TODO implement this if you want it.")

    def move(self, target_step, timeout=5):
        command = ','.join(str(item) for item in target_step)
        command = f"{Command.MOVE.value},{command}"
        logging.debug(f"Command: {command}")

        try:
          self.arduino.write(command.encode())
        except serial.SerialException as error:
          raise GripperError(f"Gripper#{self.gripper_id} failed to write to arduino {self.device_name}") from error

        self.arduino.timeout = timeout
        response = self.arduino.read_until(b'\n').decode()
        logging.debug(f"Response: {response}")

        comm_result = self.process_response(response)
        if comm_result != Response.SUCCEEDED:
          raise GripperError(f"Gripper#{self.gripper_id}: {comm_result} during move command")

        state = [int(x) for x in response.split(',')[1:]] 
        return state

    def home(self,timeout=5):
        try:
          home_pose = [512, 250, 750, 512, 250, 750, 512, 250, 750]
          return self.move(home_pose,timeout=timeout)
        except GripperError as error:
            raise GripperError(f"Gripper#{self.gripper_id}: failed to Home") from error

    def close(self):
        self.arduino.close()  