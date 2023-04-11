import logging
import backoff
import dynamixel_sdk as dxl
import time
import numpy as np
from pytimedinput import timedInput
from cares_lib.dynamixel.Servo import Servo, DynamixelServoError, ControlMode
from cares_lib.slack_bot.SlackBot import SlackBot

from configurations import GripperConfig


MOVING_STATUS_THRESHOLD = 20

# TODO extend to also take in the last action that was given to help determine how to "fix" the error
class GripperError(IOError):
    pass

def message_slack(channel, message):
    with open('slack_token.txt') as file: 
        slack_token = file.read()

    slack_bot = SlackBot(slack_token=slack_token)

    slack_bot.post_message(channel=channel, message=message)


def read_slack(gripper_id):
    with open('slack_token.txt') as file: 
        slack_token = file.read()

    slack_bot = SlackBot(slack_token=slack_token)
    
    message = slack_bot.get_message("cares-chat-bot")
    
    if message is not None:
        message = message.split(",") 
    else:
        return None

    if message[0] == str(gripper_id):
        return message[1]
    return None

def wiggle_home(gripper):
    try:
        gripper.wiggle_home()
    except GripperError as error:
        message = f"{error}"
        logging.error(message)
        message_slack("#cares-chat-bot", message)
        return
    message = f"Wiggle home result: {gripper.is_Home()}"
    logging.error(message)
    message_slack("#cares-chat-bot", message)

# TODO different handlers for the different methods - there is not one size fits all response
# TODO extend these out to do more useful stuff
def handle_gripper_error_home(gripper, error_message):
    logging.warning(f"Error handling has been initiated because of: {error_message}. Attempting to solve by home sequence.")
    
    # Try auto resolve
    gripper.wiggle_home()
    if gripper.is_home():
        return True

    # manual fix is required
    handle_gripper_error(gripper, error_message)

def handle_gripper_error(gripper, error_message):
    logging.warning(f"Error handling has been initiated because of: {error_message}")
    logging.info("Please fix the gripper and press c to try again or x to quit: ")
    message_slack("#cares-chat-bot", f"{error_message}, Please fix before the programme continues.")
    
    while True:
        value, timed_out = timedInput(timeout=10)
        if timed_out:
            value = read_slack(gripper.gripper_id)

        if value == 'c':
            logging.info("Gripper Fixed continuing onwards")
            return True
        elif value == 'x':
            logging.info("Giving up correcting gripper")
            return False
        elif value  == "wiggle" or value  == "w":
            gripper.wiggle_home()

    return True

class Gripper(object):
    def __init__(self, config : GripperConfig):

        # Setup Servor handlers
        self.gripper_id  = config.gripper_id

        self.num_motors = config.num_motors
        self.min_values = config.min_values
        self.max_values = config.max_values
        self.speed_limit = config.speed_limit
        self.velocity_min = config.velocity_min
        self.velocity_max = config.velocity_max

        self.home_sequence = config.home_sequence
        
        self.device_name = config.device_name
        self.baudrate = config.baudrate

        self.protocol = 2 # NOTE: XL-320 uses protocol 2, update if we ever use other servos

        self.port_handler   = dxl.PortHandler(self.device_name)
        self.packet_handler = dxl.PacketHandler(self.protocol)
        self.setup_handlers()

        self.group_bulk_write = dxl.GroupBulkWrite(self.port_handler, self.packet_handler)
        self.group_bulk_read  = dxl.GroupBulkRead(self.port_handler, self.packet_handler)
        
        self.servos = {}
        self.target_servo = None

        if config.actuated_target:
            try:
                self.target_servo = Servo(self.port_handler, self.packet_handler, self.protocol, config.num_motors+1, 0, config.torque_limit, config.speed_limit, 1023, 0)
            except (GripperError, DynamixelServoError) as error:
                raise GripperError(f"Gripper#{self.gripper_id}: Failed to initialise target servo") from error

        try:
            for id in range(1, self.num_motors+1):
                led = id % 7 + 1
                self.servos[id] = Servo(self.port_handler, self.packet_handler, self.protocol, id, led, config.torque_limit, config.speed_limit, self.max_values[id-1], self.min_values[id-1])
            self.setup_servos()
        except (GripperError, DynamixelServoError) as error:
            raise GripperError(f"Gripper#{self.gripper_id}: Failed to initialise servos") from error
    
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

    def setup_servos(self):
        try:
            for _, servo in self.servos.items():
                servo.enable()
            if self.target_servo is not None:
                self.target_servo.enable()
        except (GripperError, DynamixelServoError) as error:
            raise GripperError(f"Gripper#{self.gripper_id}: failed to setup servos") from error

    def ping(self):
        try:
            for _, servo in self.servos.items():
                servo.ping()
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Failed to fully Ping Gripper#{self.gripper_id}"
            if not handle_gripper_error(error_message):
                raise GripperError(error_message) from error

    # TODO handle errors
    def state(self):
        current_state = {}
        current_state["positions"]  = self.current_positions()
        current_state["velocities"] = self.current_velocity()
        current_state["loads"]      = self.current_load()
        
        current_state["object"] = None
        if self.target_servo is not None:
            current_state["object"] = self.current_object_position()

        return current_state

    def step(self):
        try:
            for _, servo in self.servos.items():
                servo.step()
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Failed to step Gripper#{self.gripper_id}"
            logging.error(error_message)
            if not handle_gripper_error(error_message):
                raise GripperError(error_message) from error

    def current_positions(self):
        try:
            current_positions = self.bulk_read("current_position", 2)
            return current_positions
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed to read current position"
            logging.error(error_message)
            raise GripperError(error_message) from error

    def current_object_position(self):
        try:
            if self.target_servo is None:
                raise ValueError("Object Servo is None")
            return self.target_servo.current_position()
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed to read object position"
            logging.error(error_message)
            raise GripperError(error_message) from error
        
    def current_velocity(self):
        try:
            current_velocity = self.bulk_read("current_velocity", 2)
            return current_velocity
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed to read current velocity"
            logging.error(error_message)
            raise GripperError(error_message) from error

    def current_load(self):
        try:
            current_load = self.bulk_read("current_load", 2)
            return current_load
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed to read current load"
            logging.error(error_message)
            raise GripperError(error_message) from error
        
    def current_control_mode(self):
        try:
            current_mode = self.bulk_read("control_mode", 1)
            return current_mode
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed to read current load"
            logging.error(error_message)
            raise GripperError(error_message) from error
    def is_moving(self):
        try:
            gripper_moving = False
            for _, servo in self.servos.items():
                gripper_moving |= servo.is_moving()
            return gripper_moving
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed to check if moving"
            logging.error(error_message)
            raise GripperError(error_message) from error

    def stop_moving(self):
        try:
            for _, servo in self.servos.items():
                servo.stop_moving()
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed to stop moving"
            logging.error(error_message)
            raise GripperError(error_message) from error

    def move_servo(self, servo_id, target_step, wait=True, timeout=5):
        if servo_id not in self.servos:
            error_message = f"Dynamixel#{servo_id} is not associated to Gripper#{self.gripper_id}"
            logging.error(error_message)
            raise ValueError(error_message)

        try:
            servo_pose = self.servos[servo_id].move(target_step, wait=wait, timeout=timeout)
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id} failed while moving Dynamixel#{servo_id}"
            logging.error(error_message)
            if not handle_gripper_error(self, error_message):
                raise GripperError(error_message) from error

    def move_servo_velocity(self, servo_id, target_velocity):
        if servo_id not in self.servos:
            error_message = (f"Dynamixel#{servo_id} is not associated to Gripper#{self.gripper_id}")
            logging.error(error_message)
            raise ValueError(error_message)
        
        try:
            servo_velocity = self.servos[servo_id].move_velocity(target_velocity)
        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id} failed while velocity moving Dynamixel#{servo_id}"
            logging.error(error_message)
            if not handle_gripper_error(self, error_message):
                raise GripperError(error_message) from error

    def move(self, steps, wait=True, timeout=5):
        if not self.verify_steps(steps):
            error_message = f"Gripper#{self.gripper_id}: The move command provided is out of bounds: Step {steps}"
            logging.error(error_message)
            raise ValueError(error_message)

        try:
            self.set_velocity(np.full(self.num_motors,self.speed_limit))
            for servo_id, servo in self.servos.items():
                
                self.set_control_mode(np.full(self.num_motors,ControlMode.JOINT.value))
                
                target_position = steps[servo_id-1]
                
                param_goal_position = [dxl.DXL_LOBYTE(target_position), dxl.DXL_HIBYTE(target_position)]
                dxl_result = self.group_bulk_write.addParam(servo_id, Servo.addresses["goal_position"] , 2, param_goal_position)

                if not dxl_result:
                    error_message = f"Gripper#{self.gripper_id}: Failed to setup move command for Dynamixel#{servo_id}"
                    logging.error(error_message)
                    raise GripperError(error_message)

            dxl_comm_result = self.group_bulk_write.txPacket()
            if dxl_comm_result != dxl.COMM_SUCCESS:
                error_message = f"Gripper#{self.gripper_id}: Failed to send move command to gripper"
                logging.error(error_message)
                raise GripperError(error_message)

            logging.debug(f"Gripper#{self.gripper_id}: Sending move command succeeded")
        
            start_time = time.perf_counter()
            while wait and self.is_moving() and time.perf_counter() < start_time + timeout:
                pass

            self.group_bulk_write.clearParam()

        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed while moving {steps}"
            logging.error(error_message)
            if not handle_gripper_error(self, error_message):
                raise GripperError(error_message) from error

    def move_velocity(self, velocities):
        if not self.verify_velocity(velocities):
            error_message = f"Gripper#{self.gripper_id}: The move velocity command provided is out of bounds velocities {velocities}"
            logging.error(error_message)
            raise ValueError(error_message)
        
        try:
            for servo_id, servo in self.servos.items():            
                
                self.set_control_mode(np.full(self.num_motors,ControlMode.WHEEL.value))
                
                target_velocity = velocities[servo_id-1]
                target_velocity_b = Servo.velocity_to_bytes(target_velocity)

                if not servo.validate_movement(target_velocity):
                    continue

                param_goal_velocity = [dxl.DXL_LOBYTE(target_velocity_b), dxl.DXL_HIBYTE(target_velocity_b)]
                dxl_result = self.group_bulk_write.addParam(servo_id, Servo.addresses["moving_speed"], 2, param_goal_velocity)

                if not dxl_result:
                    error_message = f"Gripper#{self.gripper_id}: Failed to setup move velocity command for Dynamixel#{servo_id}"
                    logging.error(error_message)
                    raise GripperError(error_message)

            dxl_comm_result = self.group_bulk_write.txPacket()
            if dxl_comm_result != dxl.COMM_SUCCESS:
                error_message = f"Gripper#{self.gripper_id}: Failed to send move velocity command to gripper"
                logging.error(error_message)
                raise GripperError(error_message)

            logging.debug(f"Gripper#{self.gripper_id}: Sending move velocity command succeeded")
            self.group_bulk_write.clearParam()

        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: Failed while moving velocity {velocities}"
            logging.error(error_message)
            if not handle_gripper_error(self, error_message):
                raise GripperError(error_message) from error
    
    def home(self):
        try:
            pose = self.home_sequence[-1]
            self.move(pose)

            if self.target_servo is not None:
                self.target_servo.move(400)#TODO abstract home position for the target servo
                self.target_servo.disable_torque()  # Need this to be target servo

            if not self.is_home():
                error_message = f"Gripper#{self.gripper_id}: failed to Home"
                logging.error(error_message)
                raise GripperError(error_message)

            logging.debug(f"Gripper#{self.gripper_id}: in home position")

        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed to Home"
            logging.error(error_message)
            if not handle_gripper_error_home(self, error_message):
                raise GripperError(error_message) from error

    # remove the back off calls to prevent stacking up on exception handling
    # @backoff.on_exception(backoff.expo, GripperError, jitter=None, giveup=handle_gripper_error)
    def wiggle_home(self):
        try:
            for pose in self.home_sequence:
                self.move(pose)

            if self.target_servo is not None:
                self.target_servo.move(400)#TODO abstract home position for the target servo
                self.target_servo.disable_torque()  # Need this to be target servo

            if not self.is_home():
                error_message = f"Gripper#{self.gripper_id}: failed to Home"
                logging.error(error_message)
                raise GripperError(error_message)

            logging.debug(f"Gripper#{self.gripper_id}: in home position")

        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: failed to Home"
            logging.error(error_message)
            if not handle_gripper_error(self, error_message):
                raise GripperError(error_message) from error
        
    def is_home(self):
        return np.all(np.abs(self.home_sequence[-1] - np.array(self.current_positions())) <= MOVING_STATUS_THRESHOLD)
    

    def set_velocity(self, velocities):   
        if not self.verify_velocity(velocities):
            error_message = f"Gripper#{self.gripper_id}: The move velocity command provided is out of bounds velocities {velocities}"
            logging.error(error_message)
            raise GripperError(error_message)

        try:     
            for servo_id, servo in self.servos.items():    
                self.set_control_mode(np.full(self.num_motors,ControlMode.JOINT.value))

                target_velocity = velocities[servo_id-1]
                target_velocity_b = Servo.velocity_to_bytes(target_velocity)

                if not servo.validate_movement(target_velocity):
                    continue

                param_goal_velocity = [dxl.DXL_LOBYTE(target_velocity_b), dxl.DXL_HIBYTE(target_velocity_b)]
                dxl_result = self.group_bulk_write.addParam(servo_id, Servo.addresses["moving_speed"], 2, param_goal_velocity)

                if not dxl_result:
                    error_message = f"Gripper#{self.gripper_id}: Failed to add move velocity param for Dynamixel#{servo_id}"
                    logging.error(error_message)
                    raise GripperError(error_message)

            dxl_comm_result = self.group_bulk_write.txPacket()
            if dxl_comm_result != dxl.COMM_SUCCESS:
                error_message = f"Gripper#{self.gripper_id}: Failed to send move velocity command to gripper"
                logging.error(error_message)
                raise GripperError(error_message)

            logging.debug(f"Gripper#{self.gripper_id}: Sending move velocity command succeeded")
            self.group_bulk_write.clearParam()

        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: Failed to change set velocity"
            logging.error(error_message)
            raise GripperError(error_message) from error
    
    def set_control_mode(self, new_mode):
        if (new_mode != self.current_control_mode()).all():
            try:
                self.disable_torque()#disable to set servo parameters

                for servo_id, servo in self.servos.items():            
                    dxl_result = self.group_bulk_write.addParam(servo_id, Servo.addresses["control_mode"], 1, [new_mode[servo_id-1]])

                    if not dxl_result:
                        error_message = f"Gripper#{self.gripper_id}: Failed to add control mode param for Dynamixel#{servo_id}"
                        logging.error(error_message)

                dxl_comm_result = self.group_bulk_write.txPacket()
                if dxl_comm_result != dxl.COMM_SUCCESS:
                    error_message = f"Gripper#{self.gripper_id}: failed to send change control mode command to gripper"
                    logging.error(error_message)
                    raise GripperError(error_message)

                logging.debug(f"Gripper#{self.gripper_id}: Change control mode command succeeded")
                self.group_bulk_write.clearParam()

                self.enable_torque()
            except (GripperError, DynamixelServoError) as error:
                error_message = f"Gripper#{self.gripper_id}: failed to change control mode"
                logging.error(error_message)
                raise GripperError(error_message) from error
             
    def enable_torque(self):
        try:
            for servo_id, servo in self.servos.items():   
                dxl_result = self.group_bulk_write.addParam(servo_id, Servo.addresses["torque_enable"], 1, [1])

                if not dxl_result:
                    error_message = f"Gripper#{self.gripper_id}: Failed to add torque enable param for Dynamixel#{servo_id}"
                    logging.error(error_message)

            dxl_comm_result = self.group_bulk_write.txPacket()
            if dxl_comm_result != dxl.COMM_SUCCESS:
                error_message = f"Gripper#{self.gripper_id}: Failed to send enable torque command to gripper"
                logging.error(error_message)
                raise GripperError(error_message)

            self.group_bulk_write.clearParam()

        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: Failed to enable torque"
            logging.error(error_message)
            raise GripperError(error_message) from error
        

    def disable_torque(self):
        try:
            for servo_id, servo in self.servos.items():            
                dxl_result = self.group_bulk_write.addParam(servo_id, Servo.addresses["torque_enable"], 1, [0])    

                if not dxl_result:
                    error_message = f"Gripper#{self.gripper_id}: Failed to add torque disable param for Dynamixel#{servo_id}"
                    logging.error(error_message)

            dxl_comm_result = self.group_bulk_write.txPacket()
            if dxl_comm_result != dxl.COMM_SUCCESS:
                error_message = f"Gripper#{self.gripper_id}: Failed to send disable torque command to gripper"
                logging.error(error_message)
                raise GripperError(error_message)

            self.group_bulk_write.clearParam()

        except (GripperError, DynamixelServoError) as error:
            error_message = f"Gripper#{self.gripper_id}: Failed to disable torque"
            logging.error(error_message)
            raise GripperError(error_message) from error

            
    def close(self):
        for _, servo in self.servos.items():
            servo.disable_torque()
        
        if self.target_servo is not None:
            self.target_servo.disable_torque()

        self.port_handler.closePort()

    def verify_steps(self, steps):
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
    
    def bulk_read(self, address, length):
        readings = []
        for id,_ in self.servos.items():
            self.bulk_read_addparam(id, address, length)

        dxl_comm_result = self.group_bulk_read.txRxPacket()
        if dxl_comm_result != dxl.COMM_SUCCESS:
            error_message = f"Gripper#{self.gripper_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            logging.error(error_message)
            raise GripperError(error_message)

        for id, _ in self.servos.items(): 
            readings.append(self.group_bulk_read.getData(id,  Servo.addresses[address], length))

        self.group_bulk_read.clearParam()  
        return readings

    def bulk_read_addparam(self, servo_id, address, length):
        dxl_addparam_result = self.group_bulk_read.addParam(servo_id, Servo.addresses[address], length)
        if not dxl_addparam_result:
            error_message = f"Gripper#{self.gripper_id} - Dynamixel#{servo_id}: groupBulkRead addparam {address} failed"
            logging.error(error_message)
            raise GripperError(error_message)

    def check_bulk_read_avaliability(self, servo_id, address, length):
        dxl_getdata_result = self.group_bulk_read.isAvailable(servo_id, Servo.addresses[address], length)
        if not dxl_getdata_result:
            error_message = f"Gripper#{self.gripper_id} - Dynamixel#{servo_id}: groupBulkRead {address} unavaliable"
            logging.error(error_message)
            raise GripperError(error_message)
        
# TODO delete this in the future...
    # @backoff.on_exception(backoff.expo, GripperError, jitter=None, giveup=handle_gripper_error)
    # def error(self):
    #     raise GripperError(f"Gripper#{self.gripper_id}: error handling test")