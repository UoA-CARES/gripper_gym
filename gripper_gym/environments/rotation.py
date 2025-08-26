import logging

from enum import Enum

import numpy as np

import gripper_gym.tools.utils as utils


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


# TODO shift this fully compositionalin future
class RotationTaskMixin:
    noise_tolerance: float
    goal_type: str

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

        if method == GOAL_SELECTION_METHOD.FIXED:
            return fixed_goals(rotator_angle, self.noise_tolerance)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_90:
            return relative_goal(1, rotator_angle)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_180:
            return relative_goal(2, rotator_angle)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_270:
            return relative_goal(3, rotator_angle)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_BETWEEN_30_330:
            return relative_goal(4, rotator_angle)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_90_180_270:
            return relative_goal_90_180_270(rotator_angle)

        # No matching goal found, throw error
        raise ValueError(f"Goal selection method unknown: {self.goal_type}")

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
        object_pose_before = previous_environment_info["poses"]["object"]
        object_pose_after = current_environment_info["poses"]["object"]

        if object_pose_after is None:
            return -1.0, False  # Penalize if object is not detected

        if object_pose_before is None:
            return 0, False  # No change in state, no reward

        yaw_before = object_pose_before["orientation"][2]
        yaw_after = object_pose_after["orientation"][2]

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
