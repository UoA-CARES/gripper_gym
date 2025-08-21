import os

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import FourFingerRotationConfig
from gripper_gym.environments.four_finger.rotation.rotation import FourFingerRotation


class FourFingerRotationFlat(FourFingerRotation):
    def __init__(
        self,
        gripper_id: int,
    ):
        env_config = FourFingerRotationConfig(gripper_id=gripper_id)

        gripper_config_path = os.path.expanduser(
            f"~/gripper_configs/{env_config.gripper_id}/gripper_config.json"
        )
        gripper_config = utils.load_gripper_config(gripper_config_path)

        super().__init__(env_config, gripper_config)

    def _reset(self):
        self.gripper.wiggle_home()
