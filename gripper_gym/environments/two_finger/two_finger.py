import logging

import cv2
from cares_lib.dynamixel.gripper_configuration import GripperConfig

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import TwoFingerConfig
from gripper_gym.environments.environment import Environment


class TwoFingerTask(Environment):
    def __init__(
        self,
        env_config: TwoFingerConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)

        # The reference position normalises the positions regardless of the camera position
        self.reference_pose = self._get_marker_poses([self.reference_marker_id])[
            self.reference_marker_id
        ]
        self.reference_position = self.reference_pose["position"]

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

    def _pose_to_state(self, pose):
        state = []
        position = pose["position"]
        state.append(position[0] - self.reference_position[0])  # X
        state.append(position[1] - self.reference_position[1])  # Y
        return state

    def _render_environment(self, state, environment_info):

        image = (
            cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180)
            if self.is_inverted
            else self.camera.get_frame()
        )

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        num_gripper_markers = self.gripper.num_motors + 2

        # account for velocity values in state
        base_index = self.gripper.num_motors if self.action_type == "velocity" else 0

        for i in range(0, num_gripper_markers):
            x = state[base_index + i * 2]
            y = state[base_index + i * 2 + 1]

            position = [
                x,
                y,
                environment_info["poses"]["gripper"][i + 1]["position"][2],
            ]

            reference = self.reference_position
            reference[2] = position[2]
            marker_pixel = utils.position_to_pixel(
                position,
                reference,
                self.camera.camera_matrix,
            )
            cv2.circle(image, marker_pixel, 9, (0, 255, 0), -1)

            cv2.putText(
                image,
                f"{i+1}",
                marker_pixel,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return image
