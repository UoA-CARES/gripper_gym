import logging
import backoff
import serial

from configurations import GripperConfig
import grippers.gripper_helper as ghlp
from cares_lib.dynamixel.Servo import DynamixelServoError

from enum import Enum

class Command(Enum):
    PING       = 0
    STOP       = 1
    MOVE       = 2
    MOVE_SERVO = 3
    GET_STATE  = 4
    LED        = 5

class Response(Enum):
    SUCCEEDED   = 0
    ERROR_STATE = 1
    TIMEOUT     = 2

class ArduinoGripper(object):
    def __init__(self, config : GripperConfig):
        # Setup Servor handlers
        self.gripper_id = config.gripper_id
        
        self.device_name = config.device_name
        self.baudrate = config.baudrate
        self.arduino = serial.Serial(config.device_name, config.baudrate)

        self.home_pose = config.home_pose

    def process_response(self, response):
      if '\n' not in response:
        #logging.debug(f"Serial Read Timeout")
        current_positions = self.current_positions()
        current_positions = [int(x) for x in current_positions]
        logging.debug(f"current positions = {current_positions}")
        return Response.TIMEOUT, current_positions
      error_state = int(response.split(',')[0])
      message = response.split(',')[1:]
      logging.debug(f"Error Flag: {error_state} {Response(error_state)} {message}")
      return Response(error_state), message

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=ghlp.handle_gripper_error)
    def send_command(self, command, timeout=5):
        try:
            self.arduino.write(command.encode())
        except serial.SerialException as error:
            raise DynamixelServoError(f"Failed to write to Gripper#{self.gripper_id} assigned port {self.device_name}") from error

        self.arduino.timeout = timeout
        response = self.arduino.read_until(b'\n').decode()
        logging.debug(f"Response: {response}")
        
        comm_result, message = self.process_response(response)
        
        if comm_result != Response.SUCCEEDED:
            if comm_result == Response.TIMEOUT: 
                state = [int(x) for x in message]
                logging.info(f"STATE: {state}")
                return state
            else:   
                raise DynamixelServoError(f"Gripper#{self.gripper_id}: {comm_result} {message}")

        state = [int(x) for x in message]
        return state

    def current_positions(self,timeout=5):
        command = f"{Command.GET_STATE.value}\n"
        logging.debug(f"Command: {command}")

        try:
            return self.send_command(command, timeout)
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to read current position") from error

    # TODO Implement GET_LOAD as a function
    # def current_load(self):
    #     command = f"{Command.GET_LOAD.value}\n"
    #     logging.debug(f"Command: {command}")

    #     try:
    #         return self.send_command(command)
    #     except DynamixelServoError as error:
    #         raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to check load") from error

    def stop_moving(self):
        command = f"{Command.STOP.value}\n"
        logging.debug(f"Command: {command}")
        try:
            return self.send_command(command)
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to stop moving") from error


    def move_servo(self, servo_id, target_step, timeout=5):
        command = f"{Command.MOVE_SERVO.value},{servo_id},{target_step}"
        logging.debug(f"Command: {command}")

        try:
            return self.send_command(command, timeout)
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Failed to move servo {servo_id} on Gripper#{self.gripper_id} to {target_step}") from error

    def move(self, target_steps, timeout=5):
        command = ','.join(str(item) for item in target_steps)
        command = f"{Command.MOVE.value},{command}"
        logging.debug(f"Command: {command}")

        try:
            return self.send_command(command, timeout)
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Failed to move Gripper#{self.gripper_id} to {target_steps}") from error
        
    def home(self,timeout=5):
        try:
          return self.move(self.home_pose,timeout=timeout)
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Failed to home Gripper#{self.gripper_id}") from error

    def ping(self):
        command = f"{Command.PING}"
        logging.debug(f"Command: {command}")
        try:
            return self.send_command(command)
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Failed to fully Ping Gripper#{self.gripper_id}") from error

    #leds, further extend to enable
    def set_leds(self):
        command = f"{Command.LED}"
        logging.debug(f"Command: {command}")
        try:
            return self.send_command(command)
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Failed to turn on LED Gripper#{self.gripper_id}") from error
        
    def close(self):
        logging.debug(f"Closing Gripper#{self.gripper_id}")
        #TODO send close command to arduino
        self.arduino.close()