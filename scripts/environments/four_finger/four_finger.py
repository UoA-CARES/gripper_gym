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
        self.iscubedropped = False
        super().__init__(env_config, gripper_config)

        self.reset_counter = 0

        # The reference position normalises the positions regardless of the camera position
        if self.env_config.reference_marker_id == 7:
            print(f"Reference Marker ID: {self.reference_marker_id}")
            self.reference_pose = self._get_marker_poses([self.reference_marker_id])[self.reference_marker_id]
            print(f"Reference Pose: {self.reference_pose}")
            self.reference_position = self.reference_pose["position"]

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _get_poses(self):
        pass

    def _get_marker_poses(self, cube_ids):
        missednum = 0
        miss_limit = 5
        reset_limit = 2
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
                self.reset_counter = 0
                break
            elif self.task == "suspended_translation" or self.task == "suspended_rotation":
                if self.iscubedropped:
                    break
                else:
                    logging.error(f"Markers not detected!")
                    self.gripper.wiggle_home()
                    time.sleep(0.5)
                    frame = cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame()
                    marker_poses = self.aruco_detector.get_marker_poses(
                        frame,
                        self.camera.camera_matrix,
                        self.camera.camera_distortion,
                        display=self.display,
                    )
                    if any(ids in list(marker_poses.keys()) for ids in cube_ids):
                        if marker_poses[list(marker_poses.keys())[0]]["position"][2] > 200:
                            logging.error(f"Cube dropped!")
                            self.iscubedropped = True
                            break
                    else:
                        input("Cube not detected! Press Enter to continue")
                        break
                    
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


        return image
    
    def _lift_reboot(self):
        pass
