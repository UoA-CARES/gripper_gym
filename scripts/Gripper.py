'''
Gripper Class intended to work with the reinforcement learning package being developed
by the University of Auckland Robotics Lab

Beth Cutler

'''
import numpy as np
import dynamixel_sdk as dxl
from Servo import Servo
from Camera import Camera

class Gripper(object):
    def __init__(self, device_name="/dev/ttyUSB0", baudrate=57600, protocol=2.0, torque_limit=180, speed_limit=100):
        # Setup Servor handlers
        self.device_name = device_name
        self.baudrate = baudrate
        self.protocol = protocol  # NOTE: XL-320 uses protocol 2

        self.port_handler = dxl.PortHandler(self.device_name)
        self.packet_handler = dxl.PacketHandler(self.protocol)

        self.group_sync_write = dxl.GroupSyncWrite(
            self.port_handler, self.packet_handler, Servo.addresses["goal_position"], 2)
        self.group_sync_read = dxl.GroupSyncRead(
            self.port_handler, self.packet_handler,  Servo.addresses["present_position"], 2)

        self.num_motors = 9
        self.servos = {}

        leds = [0, 3, 2, 0, 7, 5, 0, 4, 6]
        # Ideally these would all be the same but some are slightly physically offset
        # TODO: paramatise this further for when we have multiple grippers
        max = [900, 750, 769, 900, 750, 802, 900, 750, 794]
        min = [100,    250, 130, 100,    198, 152, 100,    250, 140]

        # create the nine servo instances
        for i in range(0, self.num_motors):
            self.servos[i] = Servo(self.port_handler, self.packet_handler,
                                   leds[i], i+1, torque_limit, speed_limit, max[i], min[i])

        # set up camera instance
        # self.camera = Camera()

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
        for _, servo in self.servos.items():
            servo.limit_torque()
            servo.limit_speed()
            servo.enable_torque()
            servo.turn_on_LED()

    """ 
    actions is an array of joint positions that specifiy the desired position of each servo
    this will be a 9 element of array of integers between 0 and 1023
    """

    def action_to_steps(self, action):
        steps = action
        for i in range(0, len(steps)):
            max = self.servos[i].max 
            min = self.servos[i].min
            steps[i] = steps[i] * (max - min) + min
        return steps

    def steps_to_angles(self, steps):
        # 0 to 1023 steps to -150 to 150 degrees 
        angles = steps
        for i in range(0,len(angles)):
            angles[i] = (angles[i] - 511.5) / 3.41 # TODO add url to documentation
        return angles

    def angles_to_steps(self, angles):
        # # -150 to 150 degrees to 0 to 1023 steps
        steps = angles
        for i in range(0,len(steps)):
            steps[i] = (3.41 * steps[i]) + 511.5 # TODO add url to documentation
        return steps

    def verify_steps(self, steps):
        # check all actions are within min max of each servo
        for id, servo in self.servos.items():
            if not servo.verify_step(steps[id]):
                print(f"Step for servo {id+1} is out of bounds")
                return False
        return True

    def is_overloaded(self):
        for _, servo in self.servos.items():
            if servo.current_load() > 20:#% of maximum load
                return True
        return False

    def move(self, steps=None, angles=None, action=None):

        terminated = False

        if angles is not None:
            steps = self.angles_to_steps(angles)

        if action is not None:
            steps = self.action_to_steps(action)

        if steps is None:
            print(f"Steps is None no action given")
            exit()

        if not self.verify_steps(steps):
            print(f"The action provided is out of bounds: {steps}")
            exit()

        for id, servo  in self.servos.items():
            self.group_sync_write.addParam(
                id+1, [dxl.DXL_LOBYTE(steps[id]), dxl.DXL_HIBYTE(steps[id])])

        # transmit the packet
        dxl_comm_result = self.group_sync_write.txPacket()
        if dxl_comm_result == dxl.COMM_SUCCESS:
            print("group_sync_write Succeeded")
        else:
            print("group_sync_write Failed")
            exit()
        self.group_sync_write.clearParam()

        # using moving flag to check if the motors have reached their goal position
        while self.gripper_moving_check():
            if self.is_overloaded():
                print("Gripper load too high")
                terminated = True
                self.close()
                self.setup()
                self.home()
                break
               
        # return the current state and whether it was terminated
        return self.current_positions(), terminated

    def current_positions(self):
        current_positions = []
        for id, servo in self.servos.items():
            current_positions.append(servo.present_position()[0])
        return current_positions

    def home(self):
        reset_seq = np.array([[512, 512],  # 1 base plate
                              [250, 512],  # 2 middle
                              [750, 512],  # 3 finger tip

                              [512, 512],  # 4 baseplate
                              [198, 460],  # 5 middle
                              [750, 512],  # 6 finger tip

                              [512, 512],  # 7 baseplate
                              [250, 512],  # 8 middle
                              [750, 512]])  # 9 finger tip

        _, gripper_error = self.move(steps=reset_seq[:, 0])
        _, gripper_error = self.move(steps=reset_seq[:, 1])
        return self.current_positions(), gripper_error

    def gripper_moving_check(self):
        moving = False
        for _, servo in self.servos.items():
            moving |= servo.moving_check()
        return moving

    def close(self):
        # disable torque
        for _, servo in self.servos.items():
            servo.disable_torque()
        # close port
        self.port_handler.closePort()
