import logging
import backoff
import dynamixel_sdk as dxl
import time

from cares_lib.dynamixel.Servo import Servo, DynamixelServoError, ControlMode
from cares_lib.slack_bot import SlackBot

from configurations import GripperConfig

def message_slack(message):
    with open('slack_token.txt') as file: 
        slack_token = file.read()

    slack_bot = SlackBot(slack_token=slack_token)

    slack_bot.post_message(channel="#cares-chat-bot", message=message)

# TODO extend these out to do more useful stuff
def handle_gripper_error(error):
    logging.warning(error)
    logging.info("Please fix the gripper and press enter to try again or x to quit: ")
    message_slack(f"{error}, please fix before the programme continues")
    value  = input()
    if value == 'x':
        logging.info("Giving up correcting gripper")
        return True 
    return False

class Gripper(object):
    def __init__(self, config : GripperConfig):

        # Setup Servor handlers
        self.gripper_id  = config.gripper_id

        self.num_motors = config.num_motors
        self.min_values = config.min_values
        self.max_values = config.max_values

        self.home_pose = config.home_pose
        
        self.device_name = config.device_name
        self.baudrate = config.baudrate

        self.protocol = 2 # NOTE: XL-320 uses protocol 2, update if we ever use other servos

        self.port_handler   = dxl.PortHandler(self.device_name)
        self.packet_handler = dxl.PacketHandler(self.protocol)
        self.setup_handlers()

        self.group_sync_write = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, Servo.addresses["goal_position"], 2)
        self.group_sync_read  = dxl.GroupSyncRead(self.port_handler, self.packet_handler, Servo.addresses["current_position"], 2)

        self.group_bulk_write = dxl.GroupBulkWrite(self.port_handler, self.packet_handler)
        self.group_bulk_read  = dxl.GroupBulkRead(self.port_handler, self.packet_handler)
        
        self.servos = {}
        self.target_servo = None

        if config.actuated_target:
            try:
                self.target_servo = Servo(self.port_handler, self.packet_handler, config.num_motors+1, 0, config.torque_limit, config.speed_limit, 1023, 0)
            except DynamixelServoError as error:
                raise DynamixelServoError(f"Gripper#{self.gripper_id}: Failed to initialise target servo") from error

        try:
            for id in range(1, self.num_motors+1):
                led = id % 7 + 1
                self.servos[id] = Servo(self.port_handler, self.packet_handler, id, led, config.torque_limit, config.speed_limit, self.max_values[id-1], self.min_values[id-1])
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

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def setup_servos(self):
        try:
            for _, servo in self.servos.items():
                servo.enable()
            
            if self.target_servo is not None:
                self.target_servo.enable()

        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to setup servos") from error

    def step(self):
        try:
            for servo_id, servo in self.servos.items():
                servo.step()
        except DynamixelServoError as error:
            self.close()
            raise DynamixelServoError(f"Failed to step Gripper#{self.gripper_id}") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def current_positions(self):
        try:
            current_positions = []
            for id, servo in self.servos.items():
                servo_position = servo.current_position()
                current_positions.append(servo_position)
            return current_positions
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to read current position") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def current_object_position(self):
        try:
            if self.target_servo is None:
                raise ValueError("Object Servo is None")
            return self.target_servo.current_position()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to read object position") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
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

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def stop_moving(self):
        try:
            for _, servo in self.servos.items():
                servo.stop_moving()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to stop moving") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
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


    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def move_servo_velocity(self, servo_id, target_velocity):
        if servo_id not in self.servos:
            error_message = (f"Dynamixel#{servo_id} is not associated to Gripper#{self.gripper_id}")
            logging.error(error_message)
            raise DynamixelServoError(error_message)
        
        try:
            servo_pose = self.servos[servo_id].move_velocity(target_velocity)
            return self.current_positions()#TODO extend to return current velocities as well
        except DynamixelServoError as error:
            self.servos[servo_id].disable_torque()
            raise DynamixelServoError(f"Gripper#{self.gripper_id} failed while velocity moving Dynamixel#{servo_id}") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def move(self, steps, wait=True, timeout=5):
        if not self.verify_steps(steps):
            error_message = f"Gripper#{self.gripper_id}: The move command provided is out of bounds: Step {steps}"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        for servo_id, servo in self.servos.items():
            servo.set_control_mode(ControlMode.JOINT.value)
            target_position = steps[servo_id-1]
            
            param_goal_position = [dxl.DXL_LOBYTE(target_position), dxl.DXL_HIBYTE(target_position)]
            dxl_result = self.group_bulk_write.addParam(servo_id, Servo.addresses["goal_position"] , 2, param_goal_position)

            if not dxl_result:
                error_message = f"Gripper#{self.gripper_id}: Failed to setup move command for Dynamixel#{servo_id}"
                logging.error(error_message)
                raise DynamixelServoError(error_message)

        dxl_comm_result = self.group_bulk_write.txPacket()
        if dxl_comm_result != dxl.COMM_SUCCESS:
            error_message = f"Gripper#{self.gripper_id}: Failed to send move command to gripper"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        logging.debug(f"Gripper#{self.gripper_id}: Sending move command succeeded")
        self.group_bulk_write.clearParam()

        try:
            start_time = time.perf_counter()
            while wait and self.is_moving() and time.perf_counter() < start_time + timeout:
                pass
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: Failed while moving") from error

        try:
            return self.current_positions()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: Failed to read its position after moving") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def move_velocity(self, velocities):        
        if not self.verify_velocity(velocities):
            error_message = f"Gripper#{self.gripper_id}: The move velocity command provided is out of bounds velocities {velocities}"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        for servo_id, servo in self.servos.items():
            servo.set_control_mode(ControlMode.WHEEL.value)

            target_velocity = velocities[servo_id-1]
            target_velocity_b = Servo.velocity_to_bytes(target_velocity)

            param_goal_velocity = [dxl.DXL_LOBYTE(target_velocity_b), dxl.DXL_HIBYTE(target_velocity_b)]
            dxl_result = self.group_bulk_write.addParam(servo_id, Servo.addresses["moving_speed"], 2, param_goal_velocity)

            if not dxl_result:
                error_message = f"Gripper#{self.gripper_id}: Failed to setup move velocity command for Dynamixel#{servo_id}"
                logging.error(error_message)

        dxl_comm_result = self.group_bulk_write.txPacket()
        if dxl_comm_result != dxl.COMM_SUCCESS:
            error_message = f"Gripper#{self.gripper_id}: Failed to send move velocity command to gripper"
            logging.error(error_message)
            raise DynamixelServoError(error_message)

        logging.debug(f"Gripper#{self.gripper_id}: Sending move velocity command succeeded")
        self.group_bulk_write.clearParam()

        try:
            return self.current_positions()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: Failed to read its position after moving") from error

    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def home(self):
        try:
            current_pose = self.move(self.home_pose)
            if self.target_servo is not None:
                self.target_servo.move(400)#TODO abstract home position for the target servo
                self.target_servo.disable_torque()  # Need this to be target servo
            return current_pose
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Gripper#{self.gripper_id}: failed to Home") from error
        
    @backoff.on_exception(backoff.expo, DynamixelServoError, jitter=None, giveup=handle_gripper_error)
    def ping(self):
        try:
            for _, servo in self.servos.items():
                servo.ping()
        except DynamixelServoError as error:
            raise DynamixelServoError(f"Failed to fully Ping Gripper#{self.gripper_id}") from error
        
    def close(self):
        # disable torque
        for _, servo in self.servos.items():
            servo.disable_torque()
        
        if self.target_servo is not None:
            self.target_servo.disable_torque()

        # close port
        self.port_handler.closePort()

    def verify_steps(self, steps):
        # check all actions are within min max of each servo
        for servo_id, servo in self.servos.items():
            step = steps[servo_id-1]
            if not servo.verify_step(step):
                logging.warn(f"Gripper#{self.gripper_id}: step for servo {servo_id} is out of bounds: {servo.min} to {servo.max}")
                return False
        return True
    
    def verify_velocity(self, velocities):
        for servo_id, servo in self.servos.items():
            velocity = velocities[servo_id-1]
            if not servo.verify_velocity(velocity):
                logging.warn(f"Gripper#{self.gripper_id}: velocity for servo {servo_id} is out of bounds")
                return False
        return True