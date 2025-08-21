import logging
import os
import time

import dynamixel_sdk as dxl
from cares_lib.dynamixel.Servo import Servo

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import FourFingerRotationSuspendedConfig
from gripper_gym.environments.four_finger.rotation.rotation import FourFingerRotation


class FourFingerRotationSuspended(FourFingerRotation):
    def __init__(
        self,
        gripper_id: int,
    ):
        env_config = FourFingerRotationSuspendedConfig(gripper_id=gripper_id)

        gripper_config_path = os.path.expanduser(
            f"~/gripper_configs/{env_config.gripper_id}/gripper_config.json"
        )
        gripper_config = utils.load_gripper_config(gripper_config_path)

        super().__init__(env_config, gripper_config)

        self.elevator_device_name = f"/dev/elevator{gripper_id}"
        self.elevator_baudrate = env_config.elevator_baudrate
        self.elevator_servo_id = env_config.elevator_servo_id

        self.elevator_max = env_config.elevator_limits[0]
        self.elevator_min = env_config.elevator_limits[1]

        self.elevator_port_handler = dxl.PortHandler(self.elevator_device_name)
        self.elevator_packet_handler = dxl.PacketHandler(2)

        self.elevator = Servo(
            self.elevator_port_handler,
            self.elevator_packet_handler,
            2,
            self.elevator_servo_id,
            1,
            200,
            200,
            self.elevator_max,
            self.elevator_min,
            model="XL330-M077-T",
        )

        self.init_elevator()

    def init_elevator(self):
        if not self.elevator_port_handler.openPort():
            error_message = f"Failed to open port {self.elevator_device_name}"
            logging.error(error_message)
            raise IOError(error_message)
        logging.debug(f"Succeeded to open port {self.elevator_device_name}")

        if not self.elevator_port_handler.setBaudRate(self.elevator_baudrate):
            error_message = f"Failed to change the baudrate to {self.elevator_baudrate}"
            logging.error(error_message)
            raise IOError(error_message)
        logging.debug(f"Succeeded to change the baudrate to {self.elevator_baudrate}")

    def _reset(self):
        self.init_elevator()
        self.elevator.enable_torque()

        self.elevator.move(self.elevator_min)  # Lower Elevator

        time.sleep(0.5)  # Let it settle

        self.gripper.wiggle_home()  # Home Gripper

        # Opening Grasp
        self.elevator.move(self.elevator_max)  # Raise Elevator

        time.sleep(0.5)  # Let it settle

        # Grasp Cube - make this a config option
        self.gripper.move(
            [2048, 2200, 2350, 2048, 2200, 2350, 2048, 2200, 2350, 2048, 2200, 2350]
        )

        time.sleep(0.5)  # Let it settle

        self.elevator.move(self.elevator_min)
