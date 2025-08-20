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

        self.goal_min = env_config.goal_min
        self.goal_max = env_config.goal_max

        self.noise_tolerance = env_config.noise_tolerance

    # overriding method
    def _choose_goal(self):
        x_min, y_min = self.goal_min
        x_max, y_max = self.goal_max

        goal_x = randrange(x_min, x_max)
        goal_y = randrange(y_min, y_max)

        return [goal_x, goal_y]

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
            state += self._pose_to_state(servo_position)

        # Object - X Y mm
        state += self._pose_to_state(environment_info["poses"]["object"])

        # Goal State - X Y mm
        state += self.goal

        # Touch Sensor Values
        if self.use_touch:
            state += environment_info["touch"]

        return state

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

        return image
