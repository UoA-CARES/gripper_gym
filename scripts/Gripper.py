import logging

import backoff
import serial
import time
import numpy as np
# from Camera import Camera

from enum import Enum, auto

class Command(Enum):
    PING       = 0
    STOP       = 1
    MOVE       = 2
    MOVE_SERVO = 3
    GET_STAT   = 4
    LED        = 5

class Response(Enum):
    SUCCEEDED   = 0
    ERROR_STATE = 1
    TIMEOUT     = 2

def handle_gripper_error(error):
    logging.warning(error)
    logging.info("Please fix the gripper and press enter to try again or x to quit: ")
    value  = input()
    if value == 'x':
        logging.info("Giving up correcting gripper")
        return True 
    elif value == 'y':
        logging.info("moving on...")
        #figure out exit condition
    return False

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
        return Response.TIMEOUT, "timed out, hopefully moving on..."
      error_state = int(response.split(',')[0])
      message = response.split(',')[1:]
      logging.debug(f"Error Flag: {error_state} {Response(error_state)} {message}")
      return Response(error_state), message

    @backoff.on_exception(backoff.expo, GripperError, jitter=None, giveup=handle_gripper_error)
    def send_command(self, command, timeout=5):
        try:
            self.arduino.write(command.encode())
        except serial.SerialException as error:
            raise GripperError(f"Failed to write to Gripper#{self.gripper_id} assigned port {self.device_name}") from error

        self.arduino.timeout = timeout
        response = self.arduino.read_until(b'\n').decode()
        logging.debug(f"Response: {response}")
        
        comm_result, message = self.process_response(response)
        
        if comm_result != Response.SUCCEEDED:
            raise GripperError(f"Gripper#{self.gripper_id}: {comm_result} {message}")

        state = [int(x) for x in message]
        return state

    def current_positions(self,timeout=5):
        command = f"{Command.GET_STATE.value}"
        logging.debug(f"Command: {command}")

        try:
            return self.send_command(command, timeout)
        except GripperError as error:
            raise GripperError(f"Failed to read position of Gripper#{self.gripper_id}") from error

        memory.add(state, action, reward)
        command = f"{Command.STOP.value}"
        logging.debug(f"Command: {command}")

        try:
            return self.send_command(command, timeout)
        except GripperError as error:
            raise GripperError(f"Failed to read stop Gripper#{self.gripper_id}") from error

    def move_servo(self, servo_id, target_step, timeout=5):
        command = f"{Command.MOVE_SERVO.value},{servo_id},{target_step}"
        logging.debug(f"Command: {command}")

        try:
            return self.send_command(command, timeout)
        except GripperError as error:
            raise GripperError(f"Failed to move servo {servo_id} on Gripper#{self.gripper_id} to {target_step}") from error

    def move(self, target_steps, timeout=5):
        command = ','.join(str(item) for item in target_steps)
        command = f"{Command.MOVE.value},{command}"
        logging.debug(f"Command: {command}")

        try:
            return self.send_command(command, timeout)
        except GripperError as error:
            raise GripperError(f"Failed to move Gripper#{self.gripper_id} to {target_steps}") from error
        
    def home(self,timeout=5):
        try:
          home_pose = [512, 250, 750, 512, 250, 750, 512, 250, 750]
          return self.move(home_pose,timeout=timeout)
        except GripperError as error:
            raise GripperError(f"Failed to home Gripper#{self.gripper_id}") from error

    def ping(self):
        command = f"{Command.PING}"
        logging.debug(f"Command: {command}")

        try:
            return self.send_command(command)
        except GripperError as error:
            raise GripperError(f"Failed to fully Ping Gripper#{self.gripper_id}") from error

    #leds, further extend to enable
    def led(self):
        
        command = f"{Command.LED}"
        logging.debug(f"Command: {command}")
        try:
            return self.send_command(command)
        except GripperError as error:
            raise GripperError(f"Failed to enable Gripper#{self.gripper_id}") from error

    def close(self):
        logging.debug(f"Closing Gripper#{self.gripper_id}")
        self.arduino.close()