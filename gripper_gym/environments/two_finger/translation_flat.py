import logging
import math
import os
import time

import dynamixel_sdk as dxl
from cares_lib.dynamixel.Servo import Servo

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import TwoFingerFlatConfig
from gripper_gym.environments.two_finger.translation import TwoFingerTranslation


class TwoFingerTranslationFlat(TwoFingerTranslation):
    def __init__(self, gripper_id: int):

        env_config = TwoFingerFlatConfig(gripper_id=gripper_id)

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

    # overriding method
    def _reset(self):
        self.init_elevator()
        self.elevator.enable_torque()
        self.elevator.set_operating_mode(4)

        # TODO implement object centred check
        self.gripper.move([312, 712, 512, 512])

        self.elevator.move(self.elevator_max)
        time.sleep(0.5)
        self.elevator.move(self.elevator_min)

        self.gripper.home()

    def _check_success(self, current_environment_info):
        target_goal = current_environment_info["goal"]

        object_current = self._pose_to_state(
            current_environment_info["poses"]["object"]
        )

        goal_distance_after = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        return goal_distance_after <= self.noise_tolerance

    def _get_poses(self):
        """
        Gets the current state of the environment using the Aruco markers.

        Returns:
        dict : A dictionary containing the poses of the gripper and object markers.

        gripper: X-Y-Z-RPY Servos + X-Y-Z-RPY Finger-tips
        object: X-Y-Z-RPY Object
        """
        poses = {}

        # Servos + Finger Tips (2)
        num_gripper_markers = self.gripper.num_motors + 2

        # Gripper markers + Object (1)
        num_markers = num_gripper_markers + 1

        # maker_ids match servo ids (counting from 1)
        marker_ids = [id for id in range(1, num_markers + 1)]

        marker_poses = self._get_marker_poses(marker_ids)

        poses["gripper"] = dict(
            [i, marker_poses[i]] for i in range(1, num_gripper_markers + 1)
        )

        # Object marker is the last one
        # This assumes that the object marker is always the last one in the list
        # and that it is not used by the gripper.
        object_marker_id = num_markers
        poses["object"] = marker_poses[object_marker_id]

        return poses

    # overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        object_previous = self._pose_to_state(
            previous_environment_info["poses"]["object"]
        )
        object_current = self._pose_to_state(
            current_environment_info["poses"]["object"]
        )
        logging.debug(
            f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}"
        )

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 80
        elif goal_distance_after > self.goal_range:
            reward = 0
        else:
            reward = round((-goal_distance_after + self.goal_range), 2)

        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        return reward, done
