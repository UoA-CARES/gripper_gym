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

    def _render_environment(self, state, environment_info):

        image = super()._render_environment(state, environment_info)

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
