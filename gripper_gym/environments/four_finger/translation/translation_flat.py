import os

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import FourFingerFlatConfig
from gripper_gym.environments.four_finger.translation.translation import (
    FourFingerTranslation,
)


class FourFingerTranslationFlat(FourFingerTranslation):
    def __init__(self, gripper_id: int):

        env_config = FourFingerFlatConfig(gripper_id=gripper_id)

        gripper_config_path = os.path.expanduser(
            f"~/gripper_configs/{env_config.gripper_id}/gripper_config.json"
        )
        gripper_config = utils.load_gripper_config(gripper_config_path)

        super().__init__(env_config, gripper_config)

    # overriding method
    def _reset(self):
        # TODO need a proper home sequence for flat translation
        self.gripper.home()
