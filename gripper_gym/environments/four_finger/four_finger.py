from cares_lib.dynamixel.gripper_configuration import GripperConfig

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import FourFingerConfig
from gripper_gym.environments.environment import Environment


class FourFingerTask(Environment):
    def __init__(
        self,
        env_config: FourFingerConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)

        self.cube_ids = env_config.cube_ids
        self.cube_size = env_config.cube_size
        self.cube_retries = env_config.cube_retries

        self.reset_counter = 0

    def _get_poses(self):
        """
        Gets the current state of the environment using the Aruco markers.

        Returns:
        dict : A dictionary containing the pose of the object marker.
        object: X-Y-Z-RPY Object
        """
        poses = {}
        # Try N times to get the object pose
        cube_pose = None
        for _ in range(0, self.cube_retries):
            marker_poses = self._get_marker_poses([])

            # TODO extract cube size and cube ids to config
            # Converts marker pose into cube pose
            cube_pose = utils.get_cube_pose(
                marker_poses,
                cube_ids=self.cube_ids,
                cube_size=self.cube_size,
            )
            if cube_pose is not None:
                break

        poses["object"] = cube_pose
        return poses

    def _render_environment(self, state, environment_info):
        # Renders base image of four_finger env
        image = super()._render_environment(state, environment_info)

        return image
