import logging
import math
from enum import Enum

import cv2
import numpy as np
from cares_lib.dynamixel.gripper_configuration import GripperConfig

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import FourFingerRotationConfig
from gripper_gym.environments.four_finger.four_finger import FourFingerTask


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


class FourFingerRotation(FourFingerTask):

    def __init__(
        self,
        env_config: FourFingerRotationConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)

        self.goal_type = env_config.goal_type
        self.noise_tolerance = env_config.noise_tolerance

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

        # Get the current object orientation
        object_pose = None
        while object_pose is None:
            poses = self._get_poses()
            object_pose = poses["object"]
            if object_pose is None:
                logging.warning("Unable to read object pose.")
                self._reset()

        object_orientation = object_pose["orientation"][2]  # Get the yaw angle

        logging.info(f"Starting Orientation: {object_orientation}")

        return self._get_goal(object_orientation)

    def _environment_info_to_state(self, environment_info):
        state = []

        # Servo Angles - Steps
        state += environment_info["gripper"]["positions"]

        # Object position - XYZ
        # Object - X Y mm
        object_pose = environment_info["poses"]["object"]
        if object_pose is None:
            # Default position if object is not detected - indicates outside of bounds
            # This is useful for tasks where the object might not be present
            # or is outside the camera's view.
            object_relative_angle = [-360]
        else:
            # Only take the yaw angle
            object_relative_angle = object_pose["orientation"][:1]

        state += object_relative_angle

        # Goal
        state += environment_info["goal"]

        # Touch Sensor Values
        if self.use_touch:
            state += environment_info["touch"]

        return [round(val, 2) for val in state]

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

    def _render_environment(self, state, environment_info):
        # Get base rendering of the four-finger environment
        image = super()._render_environment(state, environment_info)

        image = (
            cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180)
            if self.is_inverted
            else self.camera.get_frame()
        )

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        # TODO
        # Image Size X640 Y480
        position = environment_info["poses"]["object"]["position"]
        pixel_x = (
            self.camera.camera_matrix[0, 0] * position[0] / 320
            + self.camera.camera_matrix[0, 2]
        )
        pixel_y = (
            self.camera.camera_matrix[1, 1] * position[1] / 240
            + self.camera.camera_matrix[1, 2]
        )
        centre = [round(pixel_x), round(pixel_y)]

        # TODO put arrow_end calculation into function
        yaw = environment_info["poses"]["object"]["orientation"][2]
        lineSize = 35
        arrow_end_x = position[0] + (math.sin(math.radians(yaw)) * lineSize)
        arrow_end_x = (
            self.camera.camera_matrix[0, 0] * arrow_end_x / 320
            + self.camera.camera_matrix[0, 2]
        )
        arrow_end_y = position[1] - (math.cos(math.radians(yaw)) * lineSize)
        arrow_end_y = (
            self.camera.camera_matrix[1, 1] * arrow_end_y / 240
            + self.camera.camera_matrix[1, 2]
        )
        arrow_end_axis = [round(arrow_end_x), round(arrow_end_y)]

        arrow_end_x = position[0] + (math.sin(math.radians(self.goal[0])) * lineSize)
        arrow_end_x = (
            self.camera.camera_matrix[0, 0] * arrow_end_x / 320
            + self.camera.camera_matrix[0, 2]
        )
        arrow_end_y = position[1] - (math.cos(math.radians(self.goal[0])) * lineSize)
        arrow_end_y = (
            self.camera.camera_matrix[1, 1] * arrow_end_y / 240
            + self.camera.camera_matrix[1, 2]
        )
        arrow_end_goal = [round(arrow_end_x), round(arrow_end_y)]

        # Places a circle at the centre of the cube marker
        cv2.circle(image, centre, 5, (0, 0, 255), -1)
        # Draws an arrow of the markers X axis reference, this is the axis which the angle refers to. The -Y axis is seen as 0/360 degrees.
        cv2.arrowedLine(image, centre, arrow_end_axis, (255, 0, 0), 3)
        # Draws an arrow of the markers desired X axis placement, i.e. the goal angle
        cv2.arrowedLine(image, centre, arrow_end_goal, (255, 0, 0), 3)

        cv2.putText(
            image,
            f"{'Current'}",
            arrow_end_axis,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{'Goal'}",
            arrow_end_goal,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return image
