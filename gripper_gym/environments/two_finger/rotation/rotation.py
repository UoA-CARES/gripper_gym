import logging
import os
import random
from enum import Enum

import numpy as np
from cares_lib.dynamixel.Servo import Servo

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import TwoFingerRotationConfig
from gripper_gym.environments.two_finger.two_finger import TwoFingerTask


class GOAL_SELECTION_METHOD(Enum):
    FIXED = 0
    RELATIVE_90 = 1
    RELATIVE_180 = 2
    RELATIVE_270 = 3
    RELATIVE_BETWEEN_30_330 = 4
    RELATIVE_90_180_270 = 5


def fixed_goal():
    """
    Selects a random fixed goal from predefined options.
    Returns:
        int: Chosen target angle.
    """
    target_angle = np.random.randint(1, 5)
    if target_angle == 1:
        return 90
    elif target_angle == 2:
        return 180
    elif target_angle == 3:
        return 270
    elif target_angle == 4:
        return 0
    return 90


def fixed_goals(object_current_pose, noise_tolerance):
    """
    Generates fixed goals avoiding close angles.
    Args:
        object_current_pose (float): Current position of the object.
        noise_tolerance (float): Tolerance value for noise.
    Returns:
        float: Target angle.
    """

    target_angle = fixed_goal()
    while abs(object_current_pose - target_angle) < noise_tolerance:
        target_angle = fixed_goal()
    return target_angle


def relative_goal(mode, object_current_pose):

    target = 0
    if mode == 1:
        target = 90  # degrees to the right
    elif mode == 2:
        target = 180  # degrees to the right
    elif mode == 3:
        target = 270  # degrees to the right
    elif mode == 4:
        target = np.random.randint(30, 330)  # anywhere to anywhere

    return (object_current_pose + target) % 360


def relative_goal_90_180_270(object_current_pose):
    """
    Computes a relative goal based on the mode.
    Args:
        mode (int): Defines the relative angle.
        object_current_pose (float): Current position of the object.
    Returns:
        float: Computed target angle.
    """
    mode = np.random.randint(1, 4)
    logging.info(f"Target Angle Mode: {mode}")

    diff = 0
    if mode == 1:
        diff = 90  # degrees to the right
    elif mode == 2:
        diff = 180  # degrees to the right
    elif mode == 3:
        diff = 270  # degrees to the right

    current_yaw = object_current_pose
    return (current_yaw + diff) % 360


class TwoFingerRotationTask(TwoFingerTask):

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

    def _get_goal(self, rotator_angle):
        """
        Determines the goal function based on the current selection method.
        Args:
        object_state (float): Current state of the object.
        Returns:
        float: The target goal state.
        """
        # Determine which function to call based on passed in goal int value
        method = GOAL_SELECTION_METHOD[self.goal_type.upper()]

        if method == GOAL_SELECTION_METHOD.FIXED.value:
            return fixed_goals(rotator_angle, self.noise_tolerance)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_90.value:
            return relative_goal(1, rotator_angle)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_180.value:
            return relative_goal(2, rotator_angle)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_270.value:
            return relative_goal(3, rotator_angle)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_BETWEEN_30_330.value:
            return relative_goal(4, rotator_angle)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_90_180_270.value:
            return relative_goal_90_180_270(rotator_angle)

        # No matching goal found, throw error
        raise ValueError(f"Goal selection method unknown: {self.goal_type}")

    # overriding method
    def _choose_goal(self):
        """
        Chooses a goal for the current environment state.
        Returns:
        float: Chosen goal state.
        """
        # Log selected goal
        logging.info(
            f"Goal selection method = {GOAL_SELECTION_METHOD(self.goal_type.upper()).name}"
        )

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
        state += environment_info["poses"]["rotator"]

        # Goal State - angle degrees
        state += self.goal

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
        rotator_position = self.rotator.current_position()
        poses["rotator"] = round(self.rotator.step_to_angle(rotator_position))

        return poses

    # overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        """
        Computes the reward based on the target goal and the change in yaw.

        Args:
            previous_environment_info (dict): Previous state of the environment.
            current_environment_info (dict): Current state of the environment.

        Returns:
            reward: The computed reward and a boolean indicating if the task is done.
        """

        target_goal = current_environment_info["goal"]
        yaw_before = previous_environment_info["poses"]["rotator"]
        yaw_after = current_environment_info["poses"]["rotator"]

        # Compute angular error before/after (absolute shortest difference)
        goal_difference_before = utils.angular_difference(target_goal, yaw_before)
        goal_difference_after = utils.angular_difference(target_goal, yaw_after)

        # Improvement (positive if moved closer)
        delta_change = goal_difference_before - goal_difference_after

        reward = 0
        if goal_difference_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 1.0
        elif abs(delta_change) > self.noise_tolerance:
            reward = delta_change / max(goal_difference_before, 1e-6)
            reward = max(-1.0, min(1.0, reward))  # Clip reward to [-1, 1]

        return round(reward, 2), False
