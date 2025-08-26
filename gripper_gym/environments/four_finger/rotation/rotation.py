import logging
import math

import cv2
from cares_lib.dynamixel.gripper_configuration import GripperConfig

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import FourFingerRotationConfig
from gripper_gym.environments.four_finger.four_finger import FourFingerTask
from gripper_gym.environments.rotation import RotationTaskMixin


class FourFingerRotation(FourFingerTask, RotationTaskMixin):

    def __init__(
        self,
        env_config: FourFingerRotationConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)

        self.goal_type = env_config.goal_type
        self.noise_tolerance = env_config.noise_tolerance

    def _check_success(self, current_environment_info):
        target_goal = current_environment_info["goal"]

        object_pose = current_environment_info["poses"]["object"]
        if object_pose is None:
            logging.warning("Object pose is None, cannot check success.")
            return False

        current_rotation = object_pose["orientation"][2]  # Get the yaw angle

        goal_distance = utils.angular_difference(current_rotation, target_goal)

        logging.debug(f"Distance to Goal: {goal_distance}")

        return goal_distance <= self.noise_tolerance

    # overriding method
    def _choose_goal(self):
        """
        Chooses a goal for the current environment state.
        Returns:
        float: Chosen goal state.
        """
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
        previous_object_pose = self.previous_environment_info["poses"]["object"]

        if current_object_pose is None:
            logging.warning("Object pose is None, cannot render object.")
            return image

        if previous_object_pose is None:
            logging.warning(
                "Previous object pose is None, cannot render previous object."
            )
            return image

        image, current_object_pixel = utils.draw_circle(
            image,
            current_object_pose["position"],
            self.noise_tolerance,
            self.camera.camera_matrix,
            object_color,
            reference_position_mm=[0, 0, current_object_pose["position"][2]],
        )

        current_yaw = current_object_pose["orientation"][2]
        previous_yaw = previous_object_pose["orientation"][2]

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
