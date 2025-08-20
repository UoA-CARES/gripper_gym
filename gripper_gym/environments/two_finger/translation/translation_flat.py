import logging
import math
import os
import time

import cv2
import dynamixel_sdk as dxl
from cares_lib.dynamixel.Servo import Servo

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import TwoFingerFlatConfig
from gripper_gym.environments.two_finger.translation.translation import (
    TwoFingerTranslation,
)


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

        # Exclude Z for object
        object_current = self._relative_position(
            current_environment_info["poses"]["object"]
        )[:-1]

        goal_distance = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance}")

        return goal_distance <= self.noise_tolerance

    def _get_marker_poses(self, must_see_ids: list[int]) -> dict[int, dict]:
        while True:
            logging.debug(f"Attempting to Detect markers: {must_see_ids}")
            frame = (
                cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180)
                if self.is_inverted
                else self.camera.get_frame()
            )
            marker_poses = self.marker_detector.get_marker_poses(
                frame,
                self.camera.camera_matrix,
                self.camera.camera_distortion,
                display=self.display,
            )

            # This will check that all the markers are detected correctly
            if all(ids in marker_poses for ids in must_see_ids):
                break

        return marker_poses

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
    def _environment_info_to_state(self, environment_info):
        state = []

        # Servo Angles - Steps
        # state += environment_info["gripper"]["positions"]

        # Servo Velocities - Steps per second
        if self.action_type == "velocity":
            state += environment_info["gripper"]["velocities"]

        # Servo + Two Finger Tips - X Y mm
        for i in range(1, self.gripper.num_motors + 3):
            servo_position = environment_info["poses"]["gripper"][i]
            servo_relative_position = self._relative_position(servo_position)

            state += servo_relative_position[:-1]  # Exclude Z for servo tips

        # Object - X Y mm
        object_position = environment_info["poses"]["object"]
        if object_position is None:
            # Default position if object is not detected - indicates outside of bounds
            # This is useful for tasks where the object might not be present
            # or is outside the camera's view.
            object_relative_position = [-1, -1, -1]
        else:
            object_relative_position = self._relative_position(object_position)

        state += object_relative_position[:-1]  # Exclude Z for object

        # Goal State - X Y mm
        state += self.goal

        # Touch Sensor Values
        if self.use_touch:
            state += environment_info["touch"]

        return [round(val, 2) for val in state]

    # overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        # Exclude Z for object
        object_current = self._relative_position(
            current_environment_info["poses"]["object"]
        )[:-1]
        object_previous = self._relative_position(
            previous_environment_info["poses"]["object"]
        )[:-1]

        goal_difference_before = math.dist(target_goal, object_previous)
        goal_difference_after = math.dist(target_goal, object_current)

        delta_change = goal_difference_before - goal_difference_after

        reward = 0
        if goal_difference_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 1.0
        elif abs(delta_change) > self.noise_tolerance:
            reward = delta_change / max(goal_difference_before, 1e-6)
            reward = max(-1.0, min(1.0, reward))  # Clip reward to [-1, 1]

        return round(reward, 2), False
