import logging
import backoff
import dynamixel_sdk as dxl
import time

import helper as hlp
from cares_lib.dynamixel.Servo import Servo, DynamixelServoError

class U2D2Gripper(object):
    def __init__(self,
                 gripper_id=0,
                 device_name="/dev/ttyUSB1",
                 baudrate=1000000,
                 torque_limit=280,
                 speed_limit=280,
                 num_motors=4,
                 min  = [440, 260, 500, 510],
                 max  = [500, 510, 580, 760],
                 actuated_target=False):

        # Setup Servor handlers
        self.gripper_id  = gripper_id
        
        self.device_name = device_name
        self.baudrate = baudrate
        self.protocol = 2 # NOTE: XL-320 uses protocol 2, update if we ever use other servos

        self.port_handler   = dxl.PortHandler(self.device_name)
        self.packet_handler = dxl.PacketHandler(self.protocol)
        self.setup_handlers()

        self.group_sync_write = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, Servo.addresses["goal_position"], 2)
        self.group_sync_read  = dxl.GroupSyncRead(self.port_handler, self.packet_handler, Servo.addresses["current_position"], 2)

        self.servos = {}
    
        self.target_servo = None
        if actuated_target:
            self.target_servo = Servo(self.port_handler, self.packet_handler, 0, num_motors+1, 0, 1023)

        try:
            for i in range(0, num_motors):
                self.servos[i] = Servo(self.port_handler, self.packet_handler, i, i + 1, torque_limit, speed_limit, max[i], min[i])
            self.setup_servos()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: Failed to initialise servos") from error
    
    def setup_handlers(self):
        if not self.port_handler.openPort():
            error_message = f"Failed to open port {self.device_name}"
            logging.error(error_message)
            raise IOError(error_message)
        logging.info(f"Succeeded to open port {self.device_name}")

        if not self.port_handler.setBaudRate(self.baudrate):
            error_message = f"Failed to change the baudrate to {self.baudrate}"
            logging.error(error_message)
            raise IOError(error_message)
        logging.info(f"Succeeded to change the baudrate to {self.baudrate}")


    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=hlp.handle_gripper_error)
    def setup_servos(self):
        try:
            for _, servo in self.servos.items():
                servo.enable()
            
            if self.target_servo is not None:
                self.target_servo.enable()

        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to setup servos") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=hlp.handle_gripper_error)
    def current_positions(self):
        try:
            current_positions = []
            for id, servo in self.servos.items():
                servo_position = servo.current_position()
                current_positions.append(servo_position)
            return current_positions
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to read current position") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=hlp.handle_gripper_error)
    def current_load(self):
        try:
            current_load = []
            for _, servo in self.servos.items():
                current_load.append(servo.current_load())
            return current_load
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to check load") from error

    def is_moving(self):
        try:
            gripper_moving = False
            for _, servo in self.servos.items():
                gripper_moving |= servo.is_moving()
            return gripper_moving
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to check if moving") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=hlp.handle_gripper_error)
    def stop_moving(self):
        try:
            for _, servo in self.servos.items():
                servo.stop_moving()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to stop moving") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=hlp.handle_gripper_error)
    def move_servo(self, servo_id, target_step, wait=True, timeout=5):
        if servo_id not in self.servos:
            error_message = f"Dynamixel#{servo_id} is not associated to Gripper#{self.gripper_id}"
            logging.error(error_message)
            raise DynamixelServoError(error_message)
        try:
            servo_pose = self.servos[servo_id].move(target_step, wait=wait, timeout=timeout)
            return self.current_positions()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id} failed while moving Dynamixel#{servo_id}") from error


    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=hlp.handle_gripper_error)
    def move(self, steps, wait=True, timeout=5):

        if not self.verify_steps(steps):
            error_message = f"Gripper#{self.gripper_id}: The move command provided is out of bounds: Step {steps}"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        for id, servo in self.servos.items():
            servo.target_position = steps[id]
            self.group_sync_write.addParam(id + 1, [dxl.DXL_LOBYTE(steps[id]), dxl.DXL_HIBYTE(steps[id])])

        dxl_comm_result = self.group_sync_write.txPacket()
        if dxl_comm_result != dxl.COMM_SUCCESS:
            error_message = f"Gripper#{self.gripper_id}: group_sync_write Failed"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        logging.debug(f"Gripper#{self.gripper_id}: group_sync_write Succeeded")
        self.group_sync_write.clearParam()

        try:
            start_time = time.perf_counter()
            while wait and self.is_moving() and time.perf_counter() < start_time + timeout:
                pass
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed while moving") from error

        try:
            return self.current_positions()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to read its position") from error

    def home(self):
        try:
            home_pose = [440, 510, 580, 510]#TODO abstract to gripper with N servos            
            current_pose = self.move(home_pose)
            if self.target_servo is not None:
                self.target_servo.move(400)#TODO abstract home position for the target servo
            return current_pose
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to Home") from error

    def close(self):
        # disable torque
        for _, servo in self.servos.items():
            servo.disable_torque()
        
        if self.target_servo is not None:
            self.target_servo.disable_torque()

        # close port
        self.port_handler.closePort()