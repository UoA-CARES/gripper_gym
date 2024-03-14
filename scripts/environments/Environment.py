import logging
import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
from configurations import GripperEnvironmentConfig, ObjectConfig

from cares_lib.dynamixel.Gripper import Gripper
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.vision.Camera import Camera

file_path = Path(__file__).parent.resolve()


def exception_handler(error_message):
    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            try:
                return function(self, *args, **kwargs)
            except EnvironmentError as error:
                logging.error(
                    f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}"
                )
                raise EnvironmentError(
                    error.gripper,
                    f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}",
                ) from error

        return wrapper

    return decorator


class EnvironmentError(IOError):
    def __init__(self, gripper, message):
        self.gripper = gripper
        super().__init__(message)


class OBSERVATION_TYPE(Enum):
    SERVO = 0
    ARUCO = 1
    SERVO_ARUCO = 2
    IMAGE = 3


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
        object_config: ObjectConfig,
    ):
        self.task = env_config.task

        self.gripper = Gripper(gripper_config)
        self.camera = Camera(
            env_config.camera_id, env_config.camera_matrix, env_config.camera_distortion
        )

        self.observation_type = env_config.observation_type
        self.action_type = gripper_config.action_type

        self.noise_tolerance = env_config.noise_tolerance

        self.gripper.wiggle_home()

        self.aruco_detector = ArucoDetector(marker_size=env_config.marker_size)
        self.object_marker_id = object_config.object_marker_id

        self.step_counter = 0
        self.episode_horizon = env_config.episode_horizon

        # Pose to normalise the other positions against - consider (0,0)
        self.reference_marker_id = 1  # TODO make this a hyperparameter

        # The reference position normalises the positions regardless of the camera position
        self.reference_pose = self._get_marker_poses(
            [self.reference_marker_id], blindable=False
        )[self.reference_marker_id]
        self.reference_position = self.reference_pose["position"]

        object_pose = self.get_object_pose()
        self.object_state_before = self.goal_state = self._pose_to_state(object_pose)

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

        self.gripper.wiggle_home()
        state = self.get_state()

        logging.debug(f"{state}")

        # choose goal will crash if not home
        self.goal_state = self.choose_goal()

        logging.info(f"New Goal Generated: {self.goal_state}")
        return state

    def sample_action(self):
        if self.action_type == "velocity":
            return self.sample_action_velocity()
        return self.sample_action_position()

    def sample_action_position(self):
        action = []
        for i in range(0, self.gripper.num_motors):
            min_value = self.gripper.min_values[i]
            max_value = self.gripper.max_values[i]
            action.append(random.randint(min_value, max_value))
        return action

    def sample_action_velocity(self):
        action = []
        for i in range(0, self.gripper.num_motors):
            action.append(
                random.randint(self.gripper.velocity_min, self.gripper.velocity_max)
            )
        return action

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

        if self.action_type == "velocity":
            self.gripper.move_velocity_joint(action)
        else:
            self.gripper.move(action)

        state = self.get_state()
        logging.debug(f"New State: {state}")

        object_state_after = self.get_object_pose()

        logging.debug(f"New Object State: {object_state_after}")

        reward, done = self.reward_function(
            self.goal_state, self.object_state_before, object_state_after
        )

        self.object_state_before = object_state_after

        truncated = self.step_counter >= self.episode_horizon

        return state, reward, done, truncated

    @exception_handler("Failed to step gripper")
    def step_gripper(self):
        self.gripper.step()

    @exception_handler("Failed to get servo states")
    def _servo_state_space(self):
        """
        Gets the current state of the environment when using servo observations.

        Returns:
        A list representing the state of the environment: Servo Angle + X-Y-Yaw Object + Goal
        """
        state = []
        gripper_state = self.gripper.state()
        state += gripper_state["positions"]

        if self.action_type == "velocity":
            state += gripper_state["velocities"]

        state += self.get_object_pose()

        state = self.add_goal(state)

        return state

    def _pose_to_state(self, pose):
        state = []
        position = pose["position"]
        state.append(position[0] - self.reference_position[0])  # X
        state.append(position[1] - self.reference_position[1])  # Y
        return state

    def _aruco_state_space(self):
        """
        Gets the current state of the environment when using Aruco marker observations.

        Returns:
        A list representing the state of the environment.

        X-Y Servos + X-Y Finger-tips + X-Y-Yaw Object + Goal
        """
        state = []

        # Servos + Finger Tips (2) + Object (1)
        num_markers = self.gripper.num_motors + 3
        # maker_ids match servo ids (counting from 1)
        marker_ids = [id for id in range(1, num_markers + 1)]

        marker_poses = self._get_marker_poses(marker_ids, blindable=False)

        self.env_render(self.reference_position, marker_poses)

        # Add the XY poses for each of the markers in marker id into the state
        for marker_pose in marker_ids.values():
            marker_state = self._pose_to_state(marker_pose)
            state += marker_state

        if self.task == "rotation":
            # Add the additional yaw information from the object marker
            state += [marker_poses[self.object_marker_id]["orientation"][2]]  # Yaw

        # Goal State - X Y
        state = self.add_goal(state)

        return state

    @exception_handler("Failed to get servo and aruco states")
    def _servo_aruco_state_space(self):
        # Servo (Position/Velocity/Load) + Servo XY + Target XY-Yaw + Goal
        state = []
        # remove the redundent target XY-Yaw + goal from the end
        servo_state_space = self._servo_state_space()[:-4]
        # remove the goal from the end
        aruco_state_space = self._aruco_state_space()[:-1]

        state += servo_state_space
        state += aruco_state_space

        state = self.add_goal(state)

        return state

    # TODO implement function
    def _image_state_space(self):
        # Note should store the stacked frame somewhere...
        raise NotImplementedError("Requires implementation")

    @exception_handler("Failed to get state")
    def get_state(self):
        """
        Gets the current state of the environment based on the configured observation type (4 different options).

        Returns:
        A list representing the state of the environment.
        """
        if self.observation_type == OBSERVATION_TYPE.SERVO.value:
            return self._servo_state_space()
        elif self.observation_type == OBSERVATION_TYPE.ARUCO.value:
            return self._aruco_state_space()
        elif self.observation_type == OBSERVATION_TYPE.SERVO_ARUCO.value:
            return self._servo_aruco_state_space()
        elif self.observation_type == OBSERVATION_TYPE.IMAGE.value:
            return self._image_state_space()

        raise ValueError(f"Observation Type unknown: {self.observation_type}")

    def denormalize(self, action_norm):
        # return action in gripper range [-min, +max] for each servo
        action_gripper = [0 for _ in range(0, len(action_norm))]
        min_value_in = -1
        max_value_in = 1
        for i in range(0, self.gripper.num_motors):
            if self.action_type == "velocity":
                servo_min_value = self.gripper.velocity_min
                servo_max_value = self.gripper.velocity_max
            else:
                servo_min_value = self.gripper.min_values[i]
                servo_max_value = self.gripper.max_values[i]
            action_gripper[i] = int(
                (action_norm[i] - min_value_in)
                * (servo_max_value - servo_min_value)
                / (max_value_in - min_value_in)
                + servo_min_value
            )
        return action_gripper

    def normalize(self, action_gripper):
        # return action in algorithm range [-1, +1]
        max_range_value = 1
        min_range_value = -1
        action_norm = [0 for _ in range(0, len(action_gripper))]
        for i in range(0, self.gripper.num_motors):
            if self.action_type == "velocity":
                servo_min_value = self.gripper.velocity_min
                servo_max_value = self.gripper.velocity_max
            else:
                servo_min_value = self.gripper.min_values[i]
                servo_max_value = self.gripper.max_values[i]
            action_norm[i] = (action_gripper[i] - servo_min_value) * (
                max_range_value - min_range_value
            ) / (servo_max_value - servo_min_value) + min_range_value
        return action_norm

    def _get_marker_poses(self, marker_ids, blindable=False):
        while not blindable:
            logging.debug(f"Attempting to Detect markers: {marker_ids}")
            frame = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(
                frame,
                self.camera.camera_matrix,
                self.camera.camera_distortion,
                display=True,
            )

            # This will check that all the markers are detected correctly
            if all(ids in marker_poses for ids in marker_ids):
                break

        return marker_poses

    @abstractmethod
    def get_object_pose(self):
        pass

    @abstractmethod
    def ep_final_distance(self):
        pass

    @abstractmethod
    def add_goal(self, state):
        pass

    @abstractmethod
    def choose_goal(self):
        pass

    @abstractmethod
    def reward_function(self, target, start_target_pose, final_target_pose):
        pass

    # TODO shift this to a general helper/util function in cares_lib
    def _position_to_pixel(self, position, reference_position, camera_matrix):
        # pixel_n = f * N / Z + c_n
        pixel_x = (
            camera_matrix[0, 0]
            * (position[0] + reference_position[0])
            / reference_position[2]
            + camera_matrix[0, 2]
        )
        pixel_y = (
            camera_matrix[1, 1]
            * (position[1] + reference_position[1])
            / reference_position[2]
            + camera_matrix[1, 2]
        )
        return int(pixel_x), int(pixel_y)

    @abstractmethod
    def env_render(self, reference_position, marker_poses):
        pass
