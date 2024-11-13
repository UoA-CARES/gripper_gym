from abc import abstractmethod

import cv2
import logging
import tools.utils as utils
from configurations import GripperEnvironmentConfig
from environments.environment import Environment
import time

from cares_lib.dynamixel.gripper_configuration import GripperConfig


class FourFingerTask(Environment):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        self.touch_config = gripper_config.touch
        super().__init__(env_config, gripper_config)

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _get_poses(self):
        pass

    def _get_marker_poses(self, cube_ids):
        missednum = 0
        while True:
            logging.debug(f"Attempting to Detect markers: {cube_ids}")
            frame = cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(
                frame,
                self.camera.camera_matrix,
                self.camera.camera_distortion,
                display=self.display,
            )
            
            # This will check that at least one of the markers are detected correctly
            if any(ids in list(marker_poses.keys()) for ids in cube_ids):
                break
            else:
                logging.error(f"Markers not detected!")
                missednum += 1
                time.sleep(0.5)
                if missednum > 10:
                    logging.error(f"Markers not detected after 10 attempts, resetting environment.")
                    self.reset()
        return marker_poses


    @abstractmethod
    def _environment_info_to_state(self, environment_info):
        pass

    def _get_environment_info(self):
        """
        Gets the current state of the environment based on the configured observation type (4 different options).

        Returns:
        A list representing the state of the environment.
        """
        environment_info = {}
        environment_info["gripper"] = self.gripper.state()  # Gets gripper joint positions
        environment_info["poses"] = self._get_poses()       # Gets cube orientation
        environment_info["goal"] = self.goal                # Gets goal orientation

        return environment_info

    @abstractmethod
    def _choose_goal(self):
        pass

    @abstractmethod
    def _reward_function(self, previous_state, current_state, goal_state):
        pass

    def _render_environment(self, state, environment_state):
        # Renders base image of four_finger env 
        image = cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame()

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        #TODO

        return image
    
    def _lift_reboot(self):
        pass
