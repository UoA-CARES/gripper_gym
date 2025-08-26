import os

import cv2
from cares_lib.dynamixel.Servo import Servo

import gripper_gym.tools.utils as utils
from gripper_gym.configurations import TwoFingerSuspendedConfig
from gripper_gym.environments.two_finger.translation.translation import (
    TwoFingerTranslation,
)


class TwoFingerTranslationSuspended(TwoFingerTranslation):
    def __init__(self, gripper_id: int):

        env_config = TwoFingerSuspendedConfig(gripper_id=gripper_id)

        gripper_config_path = os.path.expanduser(
            f"~/gripper_configs/{env_config.gripper_id}/gripper_config.json"
        )
        gripper_config = utils.load_gripper_config(gripper_config_path)

        super().__init__(env_config, gripper_config)

        self.max_value = 3500 if gripper_config.gripper_id == 1 else 4000
        self.min_value = 0
        self.goal_line = 45
        self.bottom_line = 130  # 90 + abs(self.reference_position[1])

        self.total_moves = 0

        self.elevator_device_name = f"/dev/elevator{gripper_id}"
        self.elevator_baudrate = env_config.elevator_baudrate
        self.elevator_servo_id = env_config.elevator_servo_id

        self.elevator_max = env_config.elevator_limits[0]
        self.elevator_min = env_config.elevator_limits[1]

        self.elevator = Servo(
            self.gripper.port_handler,
            self.gripper.packet_handler,
            2,
            self.elevator_servo_id,
            1,
            200,
            200,
            self.elevator_max,
            self.elevator_min,
            model="XL330-M077-T",
        )

    # overriding method
    def _reset(self):
        self.gripper.home()
        # self._wiggle_lift()
        self._grab_cube()

    def _lift_up(self):
        self.elevator.move(self.max_value, timeout=1)

    def _lift_down(self):
        self.elevator.move(self.min_value, timeout=1)

    def _grab_cube(self):
        self._lift_up()
        # check cube is above line?
        self.gripper.move([512, 512, 362, 662])
        self._lift_down()

    def _wiggle_lift(self):
        self.elevator.move(2000, timeout=1)
        self._lift_down()
        self.elevator.move(1000, timeout=1)
        self._lift_down()

    def _get_poses(self):
        """
        Gets the current state of the environment using the Aruco markers.

        Returns:
        dict : A dictionary containing the poses of the gripper and object markers.

        gripper: X-Y-Z-RPY Servos + X-Y-Z-RPY Finger-tips
        object: X-Y-Z-RPY Object
        """
        poses = {}

        # Servos + Finger Tips (2)
        num_gripper_markers = self.gripper.num_motors + 2

        # maker_ids match servo ids (counting from 1)
        marker_ids = [id for id in range(1, num_gripper_markers + 1)]

        marker_poses = self._get_marker_poses(marker_ids)

        poses["gripper"] = dict(
            [i, marker_poses[i]] for i in range(1, num_gripper_markers + 1)
        )

        poses["object"] = utils.get_cube_pose(
            marker_poses, cube_ids=[7, 8, 9, 10, 11, 12], cube_size=50
        )

        return poses

    def _render_environment(self, state, environment_info):
        # Get base rendering of the two-finger environment translate
        image = super()._render_environment(state, environment_info)

        # Add bottom line for suspended translation task
        bounds_color = (0, 0, 255)

        _, goal_line_pixel_y = utils.position_to_pixel(
            [0, self.bottom_line], self.reference_position, self.camera.camera_matrix
        )

        cv2.line(
            image,
            (0, int(goal_line_pixel_y)),
            (640, int(goal_line_pixel_y)),
            bounds_color,
            2,
        )

        return image
