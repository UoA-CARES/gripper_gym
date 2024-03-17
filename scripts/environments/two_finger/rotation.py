import logging
import random
from enum import Enum

import numpy as np
from configurations import GripperEnvironmentConfig
from environments.two_finger.two_finger import TwoFingerTask

from cares_lib.dynamixel.gripper_configuration import GripperConfig


class REWARD_CONSTANTS(Enum):
    MAX_REWARD = 10
    MIN_REWARD = -50


class GOAL_SELECTION_METHOD(Enum):
    FIXED = 0
    RELATIVE_90 = 1
    RELATIVE_180 = 2
    RELATIVE_270 = 3
    RELATIVE_BETWEEN_30_330 = 4
    RELATIVE_90_180_270 = 5


# fixed_goal and fixed_goals functions have not been used for awhile in our relative experiments and it can already encompass them.


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

    if mode == 1:
        diff = 90  # degrees to the right
    elif mode == 2:
        diff = 180  # degrees to the right
    elif mode == 3:
        diff = 270  # degrees to the right
    elif mode == 4:
        diff = np.random.randint(30, 330)  # anywhere to anywhere

    current_yaw = object_current_pose
    return (current_yaw + diff) % 360


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
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):

        # TODO replace with different RotationEnvironment types/tasks v domains
        self.goal_selection_method = env_config.goal_selection_method

        super().__init__(env_config, gripper_config)
        self.object_observation_mode = "observed"

    def get_goal_function(self, object_state):
        """
        Determines the goal function based on the current selection method.
        Args:
        object_state (float): Current state of the object.
        Returns:
        float: The target goal state.
        """
        # Determine which function to call based on passed in goal int value
        method = self.goal_selection_method

        if method == GOAL_SELECTION_METHOD.FIXED.value:
            return fixed_goals(object_state, self.noise_tolerance)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_90.value:
            return relative_goal(1, object_state)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_180.value:
            return relative_goal(2, object_state)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_270.value:
            return relative_goal(3, object_state)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_BETWEEN_30_330.value:
            return relative_goal(4, object_state)
        elif method == GOAL_SELECTION_METHOD.RELATIVE_90_180_270.value:
            return relative_goal_90_180_270(object_state)

        # No matching goal found, throw error
        raise ValueError(f"Goal selection method unknown: {self.goal_selection_method}")

    # overriding method
    def _choose_goal(self):
        """
        Chooses a goal for the current environment state.
        Returns:
        float: Chosen goal state.
        """
        # Log selected goal
        logging.info(
            f"Goal selection method = {GOAL_SELECTION_METHOD(self.goal_selection_method).name}"
        )

        if self.object_type == "servo":
            # random home pos
            home_pos = random.randint(0, 4095)
            self.target.reset_target_servo(home_pos)

            logging.info(f"New Home Angle Generated: {self.get_home_angle(home_pos)}")

        object_state = self.get_object_pose()
        if self.object_observation_mode == "observed":
            object_state = object_state[-1]

        return self.get_goal_function(object_state)

    # overriding method
    def _reward_function(self, target_goal, yaw_before, yaw_after):
        """
        Computes the reward based on the target goal and the change in yaw.

        Args:
        target_goal: The desired goal state.
        yaw_before: The previous yaw state.
        yaw_after: The current yaw state.

        Returns:
            reward: The computed reward and a boolean indicating if the task is done.
        """
        precision_tolerance = 10

        if yaw_before is None:
            logging.debug("Start Marker Pose is None")
            return 0, True

        if yaw_after is None:
            logging.debug("Final Marker Pose is None")
            return 0, True

        done = False

        if self.object_observation_mode == "observed":
            yaw_before_rounded = round(yaw_before[-1])
            yaw_after_rounded = round(yaw_after[-1])
        elif self.object_observation_mode == "actual":
            yaw_before_rounded = round(yaw_before)
            yaw_after_rounded = round(yaw_after)

        goal_difference_before = self.rotation_min_difference(
            target_goal, yaw_before_rounded
        )
        goal_difference_after = self.rotation_min_difference(
            target_goal, yaw_after_rounded
        )

        # Current yaw_before might not equal yaw_after in prev step, hence need to check before as well to see if it has reached the goal already
        if goal_difference_before <= precision_tolerance:
            logging.info("----------Reached the Goal!----------")
            logging.info(
                "Warning: Yaw before in current step not equal to Yaw after in prev step"
            )
            reward = 10
            done = True
            return reward, done

        delta_changes = self.rotation_min_difference(
            target_goal, yaw_before_rounded
        ) - self.rotation_min_difference(target_goal, yaw_after_rounded)

        logging.info(f"Yaw = {yaw_after_rounded}")

        if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
            reward = -1
        else:
            raw_reward = delta_changes / self.rotation_min_difference(
                target_goal, yaw_before_rounded
            )
            if raw_reward >= REWARD_CONSTANTS.MAX_REWARD.value:
                reward = REWARD_CONSTANTS.MAX_REWARD.value
            elif raw_reward <= REWARD_CONSTANTS.MIN_REWARD.value:
                reward = REWARD_CONSTANTS.MIN_REWARD.value
            else:
                reward = raw_reward

        if goal_difference_after <= precision_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward += 10
            done = True

        return reward, done

    def rotation_min_difference(self, a, b):
        """
        Formula that calculates the minimum difference between two angles.

        Args:
        a: First angle.
        b: Second angle.

        Returns:
            float: The minimum angular difference.
        """
        return min(abs(a - b), (360 + min(a, b) - max(a, b)))

    def add_goal(self, state):
        """
        Adds the goal state to a given state list.

        Args:
        state (list): The list of states.

        Returns:
        list: The updated state list with the added goal state.
        """
        state.append(self.goal)
        return state

    def get_home_angle(self, home_pos):
        """
        Converts the given home position to its corresponding angle in degrees.

        Parameters:
        home_pos: The home position to be converted.

        Returns:
        Angle corresponding to the provided home position.
        """
        angle_ratio = 4096 / 360
        # 0 in decimal is 180 degrees so need to add offset
        home_pos_angle = (home_pos / angle_ratio + 180) % 360
        return home_pos_angle
