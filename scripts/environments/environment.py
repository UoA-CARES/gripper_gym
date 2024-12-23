import logging
import random
from abc import ABC, abstractmethod
from functools import wraps

import cv2
from configurations import GripperEnvironmentConfig

from cares_lib.dynamixel.Gripper import Gripper
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_lib.vision.Camera import Camera


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
        self.camera = Camera(
            env_config.camera_id, env_config.camera_matrix, env_config.camera_distortion
        )

        self.action_type = gripper_config.action_type

        self.gripper.wiggle_home()
        self.step_counter = 0
        self.goal_reward = None
        self.episode_horizon = env_config.episode_horizon

        # Pose to normalise the other positions against - consider (0,0)
        self.reference_marker_id = env_config.reference_marker_id
        
        self.goal = []
        self.current_environment_info = {}
        self.previous_environment_info = {}

    def grab_frame(self):
        frame = cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame()
        return frame

    def grab_rendered_frame(self):
        state = self._environment_info_to_state(self.current_environment_info)
        return self._render_environment(state, self.current_environment_info)

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
        return action

    def sample_action_velocity(self):
        action = []
        for _ in range(0, self.gripper.num_motors):
            action.append(
                random.randint(self.gripper.velocity_min, self.gripper.velocity_max)
            )
        return action

    def sample_action(self):
        if self.action_type == "velocity":
            return self.sample_action_velocity()
        return self.sample_action_position()

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
        
        return state, reward, done, truncated

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
    
    @exception_handler("Environment failed to reboot")
    def reboot(self):
        logging.info("Rebooting Gripper")
        self.gripper.reboot()
        self._lift_reboot()

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _get_environment_info(self):
        pass

    @abstractmethod
    def _environment_info_to_state(self, environment_info):
        pass

    @abstractmethod
    def _choose_goal(self):
        pass

    @abstractmethod
    def _reward_function(self, previous_state, current_state):
        pass

    @abstractmethod
    def _render_environment(self, state, environment_info):
        pass

    
    @abstractmethod
    def _lift_reboot(self):
        pass
