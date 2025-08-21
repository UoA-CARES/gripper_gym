import logging
import os
import random
from abc import ABC, abstractmethod
from functools import wraps

import cv2
import numpy as np
from cares_lib.dynamixel.Gripper import Gripper
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_lib.touch_sensors.sensor import SerialReader
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.vision.Camera import Camera
from cares_lib.vision.STagDetector import STagDetector

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import GripperEnvironmentConfig


# TODO rename
class TaskError(IOError):
    def __init__(self, gripper, message):
        self.gripper = gripper
        super().__init__(message)


def exception_handler(error_message):
    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            try:
                return function(self, *args, **kwargs)
            except TaskError as error:
                logging.error(
                    f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}"
                )
                raise TaskError(
                    error.gripper,
                    f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}",
                ) from error

        return wrapper

    return decorator


# TODO rename task
class Environment(ABC):
    """
    Initialise the environment with the given configurations of the gripper and object.

    Parameters:
    env_config: Configuration specific to the environment setup.
    gripper_config: Configuration specific to the gripper used.
    object_config: Configuration specific to the object in the environment.
    """

    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        self.env_config = env_config
        self.task = env_config.task
        self.domain = env_config.domain
        self.display = env_config.display

        self.gripper = Gripper(gripper_config)
        self.is_inverted = env_config.is_inverted

        camera_name = f"/dev/camera{env_config.gripper_id}"
        calibration_path = os.path.expanduser(
            f"~/gripper_configs/{env_config.gripper_id}"
        )
        camera_matrix_path = os.path.join(calibration_path, "camera_matrix.txt")
        camera_distortion_path = os.path.join(calibration_path, "camera_distortion.txt")

        self.camera = Camera(camera_name, camera_matrix_path, camera_distortion_path)

        if env_config.aruco_detector == "Aruco":
            self.marker_detector = ArucoDetector(env_config.marker_size)
        elif env_config.aruco_detector == "STag":
            self.marker_detector = STagDetector(env_config.marker_size)
        else:
            raise ValueError(
                f"Unsupported aruco_detector: {env_config.aruco_detector}. "
                "Supported values are 'Aruco' or 'STag'."
            )

        self.use_touch = env_config.use_touch

        if self.use_touch:
            touch_device_name = f"/dev/touch{env_config.gripper_id}"
            self.touch_sensor = SerialReader(touch_device_name, 921600)

            self.left_touch = False
            self.right_touch = False
            self.previous_pressure_readings = [0, 0]
            self.previous_delta_changes = [0, 0]

        self.action_type = gripper_config.action_type
        self.max_action_value = np.array(gripper_config.max_values)
        self.min_action_value = np.array(gripper_config.min_values)

        self.gripper.wiggle_home()

        self.step_counter = 0
        self.episode_horizon = env_config.episode_horizon

        # Pose to normalise the other positions against - consider (0,0)
        self.reference_marker_id = env_config.reference_marker_id

        # The reference position normalises the positions regardless of the camera position
        self.reference_pose = self._get_marker_poses([self.reference_marker_id])[
            self.reference_marker_id
        ]
        self.reference_position = self.reference_pose["position"]

        self.goal: list[int] | int = []
        self.current_environment_info: dict = {}
        self.previous_environment_info: dict = {}

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

    def _get_touch(self):
        if self.use_touch:
            return self.touch_sensor.get_latest()
        return [0, 0]

    def grab_frame(self):
        frame = (
            cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180)
            if self.is_inverted
            else self.camera.get_frame()
        )
        return frame

    def grab_rendered_frame(self):
        state = self._environment_info_to_state(self.current_environment_info)
        return self._render_environment(state, self.current_environment_info)

    def _relative_position(self, pose: dict) -> list[float]:
        relative_position = []
        position = pose["position"]
        relative_position.append(position[0] - self.reference_position[0])  # X
        relative_position.append(position[1] - self.reference_position[1])  # Y

        # This makes the reference plane 0 for Z
        relative_position.append(self.reference_position[2] - position[2])
        return relative_position

    @exception_handler("Environment failed to reset")
    def reset(self):
        """
        Resets the environment for a new episode.

        This method wiggles the gripper to its home position, generates a random home
        position (angle) for the object, chooses a new goal angle (ensuring it's not
        too close to the home angle), and resets the target servo position if necessary.

        Returns:
        list: The initial state of the environment.
        """
        self.step_counter = 0

        self._reset()

        # choose goal will crash if not home
        self.goal = self._choose_goal()
        logging.debug(f"New Goal Generated: {self.goal}")

        self.previous_environment_info = self.current_environment_info = (
            self._get_environment_info()
        )
        logging.debug(f"Env Info: {self.current_environment_info}")

        state = self._environment_info_to_state(self.current_environment_info)
        logging.debug(f"State: {state}")

        return state

    def sample_action_position(self):
        action = []
        for i in range(0, self.gripper.num_motors):
            min_value = self.gripper.min_values[i]
            max_value = self.gripper.max_values[i]
            action.append(random.randint(min_value, max_value))
        return np.array(action)

    def sample_action_velocity(self):
        action = []
        for _ in range(0, self.gripper.num_motors):
            action.append(
                random.randint(self.gripper.velocity_min, self.gripper.velocity_max)
            )
        return np.array(action)

    def sample_action(self):
        if self.action_type == "velocity":
            return self.sample_action_velocity()
        return self.sample_action_position()

    def _get_environment_info(self):
        """
        Gets the current state of the environment based on the configured observation type (4 different options).

        Returns:
        A list representing the state of the environment.
        """
        environment_info = {}
        environment_info["gripper"] = self.gripper.state()
        environment_info["poses"] = self._get_poses()
        environment_info["touch"] = self._get_touch()
        environment_info["goal"] = self.goal
        environment_info["success"] = self._check_success(environment_info)

        return environment_info

    @exception_handler("Failed to step")
    def step(self, action):
        """
        Takes a step in the environment using the given action and returns the results.

        Parameters:
        action: The action to be executed.

        Returns:
        state: The new state after executing the action.
        reward: The reward obtained after the action.
        done: Whether the episode is done or not.
        truncated: Whether the step was truncated or not.
        """
        self.step_counter += 1

        action = np.round(np.asarray(action)).astype(
            int
        )  # Convert to int, as sample_action returns float

        if self.action_type == "velocity":
            self.gripper.move_velocity_joint(action)
        else:
            self.gripper.move(action)

        self.current_environment_info = self._get_environment_info()
        state = self._environment_info_to_state(self.current_environment_info)
        image = self._render_environment(state, self.current_environment_info)

        if self.display:
            cv2.imshow("State Image", image)
            cv2.waitKey(10)

        reward, done = self._reward_function(
            self.previous_environment_info, self.current_environment_info
        )

        self.previous_environment_info = self.current_environment_info

        truncated = self.step_counter >= self.episode_horizon

        return state, reward, done, truncated, self.current_environment_info

    @exception_handler("Environment failed to reboot")
    def reboot(self):
        logging.info("Rebooting Gripper")
        self.gripper.reboot()

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _environment_info_to_state(self, environment_info):
        pass

    @abstractmethod
    def _choose_goal(self):
        pass

    @abstractmethod
    def _reward_function(self, previous_environment_info, current_environment_info):
        pass

    @abstractmethod
    def _render_environment(self, state, environment_info):
        image = (
            cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180)
            if self.is_inverted
            else self.camera.get_frame()
        )

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        # Draw reference marker
        reference_pixels = utils.position_to_pixel(
            [0, 0, 0], self.reference_position, self.camera.camera_matrix
        )
        cv2.circle(image, reference_pixels, 5, (255, 0, 0), -1)

        cv2.putText(
            image,
            "(0,0,0)",
            reference_pixels,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return image

    # TODO rename
    @abstractmethod
    def _get_poses(self):
        pass

    @abstractmethod
    def _check_success(self, current_environment_info):
        pass
