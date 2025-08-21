import logging
import math
from random import randrange

import cv2
from cares_lib.dynamixel.gripper_configuration import GripperConfig

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import TwoFingerTranslationConfig
from gripper_gym.environments.two_finger.two_finger import TwoFingerTask


class TwoFingerTranslation(TwoFingerTask):
    def __init__(
        self,
        env_config: TwoFingerTranslationConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)

        self.reward_function = env_config.reward_function

        self.goal_min = env_config.goal_min
        self.goal_max = env_config.goal_max

        self.goal_range = env_config.goal_range

        self.noise_tolerance = env_config.noise_tolerance

    # overriding method
    def _choose_goal(self):
        x_min, y_min = self.goal_min
        x_max, y_max = self.goal_max

        goal_x = randrange(x_min, x_max)
        goal_y = randrange(y_min, y_max)

        return [goal_x, goal_y]

    def _check_success(self, current_environment_info):
        target_goal = current_environment_info["goal"]

        # Exclude Z for object
        object_current = self._relative_position(
            current_environment_info["poses"]["object"]
        )[:-1]

        goal_distance = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance}")

        return goal_distance <= self.noise_tolerance

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

            state += servo_relative_position[:-1]  # Exclude Z for servo position

        # Object - X Y mm
        object_pose = environment_info["poses"]["object"]
        if object_pose is None:
            # Default position if object is not detected - indicates outside of bounds
            # This is useful for tasks where the object might not be present
            # or is outside the camera's view.
            object_relative_position = [-1, -1]
        else:
            object_relative_position = self._relative_position(object_pose)

        state += object_relative_position[:-1]  # Exclude Z for object

        # Goal State - X Y mm
        state += self.goal

        # Touch Sensor Values
        if self.use_touch:
            state += environment_info["touch"]

        return [round(val, 2) for val in state]

    # overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        match self.reward_function:
            case "delta":
                return self._reward_function_delta(
                    previous_environment_info, current_environment_info
                )
            case "distance":
                return self._reward_function_linear(
                    previous_environment_info, current_environment_info
                )

        return self._reward_function_delta(
            previous_environment_info, current_environment_info
        )

    def _reward_function_delta(
        self, previous_environment_info, current_environment_info
    ):
        reward = 0

        target_goal = current_environment_info["goal"]

        object_pose_current = current_environment_info["poses"]["object"]
        object_pose_previous = previous_environment_info["poses"]["object"]

        if object_pose_current is None:
            # If current pose is None, we cannot compute a delta change
            return 0.0, False

        if object_pose_previous is None:
            # If previous pose is None, we cannot compute a delta change
            return 0.0, False

        # This now converts the poses with respect to reference marker
        # Exclude Z for object
        object_current = self._relative_position(object_pose_current)[:-1]
        object_previous = self._relative_position(object_pose_previous)[:-1]

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

    def _reward_function_linear(
        self, previous_environment_info, current_environment_info
    ):
        target_goal = current_environment_info["goal"]

        object_pose_current = current_environment_info["poses"]["object"]

        # This now converts the poses with respect to reference marker
        # Exclude Z for object
        if object_pose_current is not None:
            object_current = self._relative_position(object_pose_current)[:-1]
            goal_distance_after = math.dist(target_goal, object_current)
        else:
            # Set to a value outside the goal range
            goal_distance_after = self.goal_range + self.noise_tolerance

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        reward = 0
        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 1.0
        else:
            reward = max(
                0.0,
                1.0
                - (goal_distance_after - self.noise_tolerance)
                / (self.goal_range - self.noise_tolerance),
            )

        return round(reward, 2), False

    def _render_environment(self, state, environment_info):
        # Get base rendering of the four-finger environment
        image = super()._render_environment(state, environment_info)

        # Draw the goal boundry for the translation task
        bounds_color = (255, 0, 0)
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

        # Draw object positions
        object_color = (0, 255, 0)

        noise_tolerance_pixels = utils.mm_to_pixels(
            self.noise_tolerance, self.reference_position[2], self.camera.camera_matrix
        )
        noise_tolerance_pixels = int(noise_tolerance_pixels)

        # Draw object's current position
        current_object_pose = environment_info["poses"]["object"]
        if current_object_pose is not None:
            image, current_object_pixel = utils.draw_circle(
                image,
                current_object_pose["position"],
                self.noise_tolerance,
                self.camera.camera_matrix,
                object_color,
                reference_position_mm=[0, 0, current_object_pose["position"][2]],
            )

            cv2.putText(
                image,
                "Current",
                (
                    current_object_pixel[0] + noise_tolerance_pixels,
                    current_object_pixel[1] + noise_tolerance_pixels,
                ),  # Text location adjusted for circle size
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                object_color,
                2,
            )

        # Draw object's previous position
        previous_object_pose = self.previous_environment_info["poses"]["object"]
        if previous_object_pose is not None:
            image, previous_object_pixel = utils.draw_circle(
                image,
                previous_object_pose["position"][0:2],
                self.noise_tolerance,
                self.camera.camera_matrix,
                object_color,
                reference_position_mm=[0, 0, previous_object_pose["position"][2]],
            )

            cv2.putText(
                image,
                "Previous",
                (
                    previous_object_pixel[0] + noise_tolerance_pixels,
                    previous_object_pixel[1] + noise_tolerance_pixels,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                object_color,
                2,
            )

        # Draw line from previous to current
        if current_object_pose is not None and previous_object_pose is not None:
            cv2.line(image, current_object_pixel, previous_object_pixel, (255, 0, 0), 2)

        # Draw goal position - note the reference Z is relative to the Marker ID of the target for proper math purposes
        goal_color = (0, 0, 255)
        image, goal_pixel = utils.draw_circle(
            image,
            self.goal,
            self.noise_tolerance,
            self.camera.camera_matrix,
            goal_color,
            reference_position_mm=self.reference_position,
        )

        # Draw line from object to goal
        cv2.line(image, current_object_pixel, goal_pixel, (255, 0, 0), 2)

        reward, _ = self._reward_function(
            self.previous_environment_info, self.current_environment_info
        )
        cv2.putText(
            image,
            f"Reward: {reward}",
            (
                goal_pixel[0] + noise_tolerance_pixels,
                goal_pixel[1] + noise_tolerance_pixels,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        # Draw goal range circle if using distance reward function
        if self.reward_function == "distance":
            goal_range_color = (0, 255, 0)
            goal_range_pixels = utils.mm_to_pixels(
                self.goal_range, self.reference_position[2], self.camera.camera_matrix
            )
            goal_range_pixels = int(goal_range_pixels)

            # Draw goal range circle
            cv2.circle(
                image,
                goal_pixel,
                goal_range_pixels,
                goal_range_color,
                2,
            )

        return image
