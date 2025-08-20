import logging
import time

import cv2
from cares_lib.dynamixel.gripper_configuration import GripperConfig

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import GripperEnvironmentConfig
from gripper_gym.environments.environment import Environment


class FourFingerTask(Environment):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)

        self.is_cubed_dropped = False

        self.reset_counter = 0

    def _get_marker_poses(self, cube_ids):
        while True:
            logging.debug(f"Attempting to Detect markers: {cube_ids}")
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

            # This will check that at least one of the markers are detected correctly
            if any(ids in list(marker_poses.keys()) for ids in cube_ids):
                self.reset_counter = 0
                break
            elif (
                self.task == "suspended_translation"
                or self.task == "suspended_rotation"
            ):
                if self.is_cubed_dropped:
                    break
                else:
                    logging.error(f"Markers not detected!")
                    self.gripper.wiggle_home()
                    time.sleep(0.5)
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
                    if any(ids in list(marker_poses.keys()) for ids in cube_ids):
                        if (
                            marker_poses[list(marker_poses.keys())[0]]["position"][2]
                            > 200
                        ):
                            logging.error(f"Cube dropped!")
                            self.is_cubed_dropped = True
                            break
                    else:
                        input("Cube not detected! Press Enter to continue")
                        break

        return marker_poses

    def _get_poses(self):
        """
        Gets the current state of the environment using the Aruco markers.

        Returns:
        dict : A dictionary containing the pose of the object marker.
        object: X-Y-Z-RPY Object
        """
        poses = {}
        marker_poses = self._get_marker_poses(self.env_config.cube_ids)

        # TODO extract cube size and cube ids to config
        # Converts marker pose into cube pose
        poses["object"] = utils.get_cube_pose(
            marker_poses, cube_ids=[1, 2, 3, 4, 5, 6], cube_size=50
        )

        return poses

    def _render_environment(self, state, environment_info):
        # Renders base image of four_finger env
        image = super()._render_environment(state, environment_info)

        return image
