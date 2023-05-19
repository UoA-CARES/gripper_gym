import logging
import time
import random
import threading
from enum import Enum
from queue import Queue, Empty
from time import sleep
import numpy as np

from serial import Serial

from configurations import ObjectConfig

from cares_lib.dynamixel.Servo import Servo, DynamixelServoError, ControlMode

class Command(Enum):
    GET_YAW = 0
    OFFSET = 1

class MagnetObject(object):
    def __init__(self, config : ObjectConfig, aruco_yaw) -> None:
        self.serial = Serial(config.device_name, config.baudrate)
        sleep(1)

        self.offset(aruco_yaw)

    def get_response(self):
        return self.serial.read_until(b'\n').decode().split(",")
    
    def get_yaw(self):
        command = f"{Command.GET_YAW.value},"
        try:
            self.serial.write(command.encode())
            response = self.get_response()

            while response[0] != "YAW":
                self.serial.write(command.encode())
                response = self.get_response()

            yaw = float(response[1])
            return yaw
        except (UnicodeDecodeError, ValueError):
            logging.info("get_yaw: Error reading from serial port, retrying...")

    def offset(self, aruco_yaw):
        logging.info("Calibrating magnet to aruco reading...")
        command = f"{Command.OFFSET.value},{aruco_yaw}\n"
        try:
            self.serial.write(command.encode())
            response = self.get_response()

            while response[0] != "OFFSET":
                self.serial.write(command.encode())
                response = self.get_response()

            success = bool(response[1])
            return success
        except (UnicodeDecodeError, ValueError):
            logging.info("Offset: Error reading from serial port, retrying...")

    
    
    def reset(self):
        pass

class ServoObject(object):
    def __init__(self, port_handler, packet_handler, servo_id, model) -> None:
        self.model = model

        self.min = 0
        self.max = 4094 if self.model == "XL430-W250-T" else 1023
        self.target_servo = Servo(port_handler, packet_handler, 2.0, servo_id, 0, 200, 200, self.max, self.min, self.model)

    def get_yaw(self):
        current_position = self.target_servo.current_position()
        yaw = self.target_servo.step_to_angle(current_position)
        if yaw < 0:
            yaw += 360
        return yaw
    
    def reset(self):
        reset_home_position = random.randint(self.min, self.max)
        self.target_servo.move(reset_home_position)
        self.target_servo.disable_torque()