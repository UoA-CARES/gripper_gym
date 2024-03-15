import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from random import randrange

import cv2
import numpy as np
from environments.Environment import Environment

file_path = Path(__file__).parent.resolve()
import tools.utils as utils
from configurations import GripperEnvironmentConfig, ObjectConfig

from cares_lib.dynamixel.gripper_configuration import GripperConfig


class TwoFingerTask(Environment):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
        object_config: ObjectConfig,
    ):
        super().__init__(env_config, gripper_config, object_config)

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

        marker_poses = self._get_marker_poses(marker_ids, blindable=False)

        poses["gripper"] = dict(
            [i, marker_poses[i]] for i in range(1, num_gripper_markers + 1)
        )
        poses["object"] = marker_poses[self.object_marker_id]

        return poses

    @abstractmethod
    def _environment_info_to_state(self, environment_info):
        pass

    def _get_environment_info(self):
        """
        Gets the current state of the environment based on the configured observation type (4 different options).

        Returns:
        A list representing the state of the environment.
        """
        environment_info = {}
        environment_info["gripper"] = self.gripper.state()
        environment_info["poses"] = self._get_poses()
        environment_info["goal"] = self.goal

        return environment_info

    @abstractmethod
    def _choose_goal(self):
        pass

    @abstractmethod
    def _reward_function(self, previous_state, current_state, goal_state):
        pass

    def _pose_to_state(self, pose):
        state = []
        position = pose["position"]
        state.append(position[0] - self.reference_position[0])  # X
        state.append(position[1] - self.reference_position[1])  # Y
        return state

    def _render_envrionment(self, state, environment_state):

        image = self.camera.get_frame()

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        # Draw the goal boundry
        bounds_color = (0, 255, 0)
        bounds_min_x, bounds_min_y = utils.position_to_pixel(
            self.goal_min, self.reference_position, self.camera.camera_matrix
        )
        bounds_max_x, bounds_max_y = utils.position_to_pixel(
            self.goal_max, self.reference_position, self.camera.camera_matrix
        )
        cv2.rectangle(
            image,
            (int(bounds_min_x), int(bounds_min_y)),
            (int(bounds_max_x), int(bounds_max_y)),
            bounds_color,
            2,
        )

        # Draw object position
        object_color = (0, 255, 0)
        object_pose = environment_state["poses"]["object"]
        object_pixel = utils.position_to_pixel(
            object_pose["position"],
            [0, 0, object_pose["position"][2]],
            self.camera.camera_matrix,
        )
        cv2.circle(image, object_pixel, 9, object_color, -1)

        # Draw goal position - note the reference Z is relative to the Marker ID of the target for proper math purposes
        goal_color = (0, 0, 255)
        goal_reference_position = [
            self.reference_position[0],
            self.reference_position[1],
            object_pose["position"][2],
        ]
        goal_pixel = utils.position_to_pixel(
            self.goal, goal_reference_position, self.camera.camera_matrix
        )
        cv2.circle(image, goal_pixel, 9, goal_color, -1)

        num_gripper_markers = self.gripper.num_motors + 2

        # account for servo position and velocity values in state
        base_index = self.gripper.num_motors + (self.gripper.num_motors if self.action_type == "velocity" else 0)
        for i in range(0, num_gripper_markers):
            x = state[base_index + i * 2]
            y = state[base_index + i * 2 + 1]

            position = [
                x,
                y,
                environment_state["poses"]["gripper"][i + 1]["position"][2],
            ]

            reference = self.reference_position
            reference[2] = position[2]
            marker_pixel = utils.position_to_pixel(
                position,
                reference,
                self.camera.camera_matrix,
            )
            cv2.circle(image, marker_pixel, 9, (0, 255, 0), -1)

            cv2.putText(
                image,
                f"{i+1}",
                marker_pixel,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("State Image", image)
        cv2.waitKey(10)


class TwoFingerTranslation(TwoFingerTask):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
        object_config: ObjectConfig,
    ):
        self.noise_tolerance = 15

        # These bounds are respective to the reference marker in Environment
        self.goal_min = [-30.0, 60.0]
        self.goal_max = [120.0, 110.0]
        logging.info(f"Goal Max: {self.goal_max}")
        logging.info(f"Goal Min: {self.goal_min}")

        super().__init__(env_config, gripper_config, object_config)

    # overriding method
    def _choose_goal(self):
        x1, y1 = self.goal_min
        x2, y2 = self.goal_max

        goal_x = randrange(x1, x2)
        goal_y = randrange(y1, y2)

        logging.info(f"New Goal: {goal_x}, {goal_y}")
        return [goal_x, goal_y]

    def _environment_info_to_state(self, environment_info):
        state = []

        # Servo Angles - Steps
        state += environment_info["gripper"]["positions"]

        # Servo Velocities - Steps per second
        if self.action_type == "velocity":
            state += environment_info["gripper"]["velocities"]

        # Servo + Two Finger Tips - X Y mm
        for i in range(1, self.gripper.num_motors + 3):
            servo_position = environment_info["poses"]["gripper"][i]
            state += self._pose_to_state(servo_position)

        # Object - X Y mm
        state += self._pose_to_state(environment_info["poses"]["object"])

        # Goal State - X Y mm
        state += self.goal

        return state

    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        object_previous = previous_environment_info["poses"]["object"]["position"][0:2] # X Y
        object_current = current_environment_info["poses"]["object"]["position"][0:2] # X Y

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        goal_progress = goal_distance_before - goal_distance_after

        # The following step might improve the performance.

        # goal_before_array = goal_before[0:2]
        # delta_changes   = np.linalg.norm(target_goal - goal_before_array) - np.linalg.norm(target_goal - goal_after_array)
        # if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
        #     reward = -10
        # else:
        #     reward = -goal_difference
        #     #reward = delta_changes / (np.abs(yaw_before - target_goal))
        #     #reward = reward if reward > 0 else 0

        # For Translation. noise_tolerance is 15, it would affect the performance to some extent.
        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            done = True
            reward = 500
        else:
            reward += goal_progress

        logging.info(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        return reward, done
