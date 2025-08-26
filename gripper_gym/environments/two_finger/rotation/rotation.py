import logging
import math
import os
import random

import cv2
from cares_lib.dynamixel.Servo import Servo

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import TwoFingerRotationConfig
from gripper_gym.environments.rotation import RotationTaskMixin
from gripper_gym.environments.two_finger.two_finger import TwoFingerTask


class TwoFingerRotation(TwoFingerTask, RotationTaskMixin):

    def __init__(
        self,
        gripper_id: int,
    ):
        env_config = TwoFingerRotationConfig(gripper_id=gripper_id)

        gripper_config_path = os.path.expanduser(
            f"~/gripper_configs/{env_config.gripper_id}/gripper_config.json"
        )
        gripper_config = utils.load_gripper_config(gripper_config_path)

        super().__init__(env_config, gripper_config)

        self.goal_type = env_config.goal_type
        self.noise_tolerance = env_config.noise_tolerance

        self.rotator_baudrate = env_config.rotator_baudrate
        self.rotator_servo_id = env_config.rotator_servo_id

        self.rotator = Servo(
            self.gripper.port_handler,
            self.gripper.packet_handler,
            2,
            self.rotator_servo_id,
            1,
            200,
            200,
            4095,
            0,
            model="XL330-M077-T",
        )

    def _check_success(self, current_environment_info):
        target_goal = current_environment_info["goal"]

        # Exclude Z for object
        current_rotation = current_environment_info["poses"]["rotator"]

        goal_distance = utils.angular_difference(current_rotation, target_goal)

        logging.debug(f"Distance to Goal: {goal_distance}")

        return goal_distance <= self.noise_tolerance

    def _reset(self):
        self.gripper.home()

    # overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        self._reward_function(previous_environment_info, current_environment_info)

    # overriding method
    def _choose_goal(self):
        """
        Chooses a goal for the current environment state.
        Returns:
        float: Chosen goal state.
        """

        rotator_steps = random.randint(0, 4095)
        self.rotator.move(rotator_steps)

        rotator_angle = self.rotator.step_to_angle(rotator_steps)
        logging.info(f"New Home Angle Generated: {rotator_angle}")

        return self._get_goal(rotator_angle)

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

        # Rotator - angle degrees
        state += [environment_info["poses"]["rotator"]]

        # Goal State - angle degrees
        state += [self.goal]

        # Touch Sensor Values
        if self.use_touch:
            state += environment_info["touch"]

        return state

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

        # Rotator position - get actual position in degrees from the servo
        rotator_position = self.rotator.current_position()
        poses["rotator"] = round(self.rotator.step_to_angle(rotator_position))

        return poses

    def _render_environment(self, state, environment_info):
        # Get base rendering of the four-finger environment
        image = super()._render_environment(state, environment_info)

        noise_tolerance_pixels = utils.mm_to_pixels(
            self.noise_tolerance, self.reference_position[2], self.camera.camera_matrix
        )
        noise_tolerance_pixels = int(noise_tolerance_pixels)

        # Draw object position
        object_color = (0, 255, 0)

        current_object_pose = environment_info["poses"]["object"]
        image, current_object_pixel = utils.draw_circle(
            image,
            current_object_pose["position"],
            self.noise_tolerance,
            self.camera.camera_matrix,
            object_color,
            reference_position_mm=[0, 0, current_object_pose["position"][2]],
        )

        current_yaw = environment_info["poses"]["rotator"]
        previous_yaw = self.previous_environment_info["poses"]["rotator"]

        line_length = 50  # Length of the arrow lines in pixels

        # Calculate the end points of the arrows
        current_arrow_x = int(
            current_object_pixel[0]
            + (math.sin(math.radians(current_yaw)) * line_length)
        )
        current_arrow_y = int(
            current_object_pixel[1]
            - (math.cos(math.radians(current_yaw)) * line_length)
        )

        previous_arrow_x = int(
            current_object_pixel[0]
            + (math.sin(math.radians(previous_yaw)) * line_length)
        )
        previous_arrow_y = int(
            current_object_pixel[1]
            - (math.cos(math.radians(previous_yaw)) * line_length)
        )

        goal_arrow_x = int(
            current_object_pixel[0] + (math.sin(math.radians(self.goal)) * line_length)
        )
        goal_arrow_y = int(
            current_object_pixel[1] - (math.cos(math.radians(self.goal)) * line_length)
        )

        # Draws an arrow of the markers X axis reference, this is the axis which the angle refers to. The -Y axis is seen as 0/360 degrees.
        cv2.arrowedLine(
            image,
            current_object_pixel,
            (current_arrow_x, current_arrow_y),
            (255, 0, 0),
            3,
        )
        # Draws an arrow of the markers desired X axis placement, i.e. the goal angle
        cv2.arrowedLine(
            image,
            current_object_pixel,
            (previous_arrow_x, previous_arrow_y),
            (0, 255, 0),
            3,
        )

        cv2.arrowedLine(
            image,
            current_object_pixel,
            (goal_arrow_x, goal_arrow_y),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            image,
            f"{'Current'}",
            (current_arrow_x, current_arrow_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{'Previous'}",
            (previous_arrow_x, previous_arrow_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{'Goal'}",
            (goal_arrow_x, goal_arrow_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return image
