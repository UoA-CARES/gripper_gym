import logging
import math
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

        self.reward_function = env_config.reward_function

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

    def _reward_function(self, previous_environment_info, current_environment_info):
        match self.reward_function:
            case "distance":
                return self._reward_function_distance(
                    previous_environment_info, current_environment_info
                )
            case "delta_change":
                return self._reward_function_delta_change(
                    previous_environment_info, current_environment_info
                )
            case "staged":
                return self._reward_function_staged(
                    previous_environment_info, current_environment_info
                )
            case "touch_staged":
                return self._reward_function_touch_staged(
                    previous_environment_info, current_environment_info
                )
            case _:
                return self._reward_function_staged(
                    previous_environment_info, current_environment_info
                )

    # overriding method
    def _reward_function_distance(
        self, previous_environment_info, current_environment_info
    ):
        self.goal_range = 50
        self.goal_reward = 60
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        # Exclude Z for object
        object_previous = self._relative_position(
            previous_environment_info["poses"]["object"]
        )[:-1]
        object_current = self._relative_position(
            current_environment_info["poses"]["object"]
        )[:-1]
        logging.debug(
            f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}"
        )

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = self.goal_reward
        elif (
            goal_distance_after > self.goal_range
            or object_current[1] >= self.bottom_line
        ):
            reward = 0
        else:
            reward = round((-goal_distance_after + self.goal_range), 2)

        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        print(object_current, reward)

        return reward, done

    # overriding method
    def _reward_function_delta_change(
        self, previous_environment_info, current_environment_info
    ):
        self.goal_range = 25
        self.goal_reward = 4  # goal reward minus the potential moving away negativity
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        object_previous = self._relative_position(
            previous_environment_info["poses"]["object"]
        )[:-1]
        object_current = self._relative_position(
            current_environment_info["poses"]["object"]
        )[:-1]
        logging.debug(
            f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}"
        )

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        if object_current[1] <= self.bottom_line:
            reward = 1

            delta_changes = goal_distance_before - goal_distance_after

            raw_reward = delta_changes / goal_distance_before

            reward += 1 if raw_reward >= 1 else -1 if raw_reward <= -1 else raw_reward

            if goal_distance_after <= self.goal_range:
                logging.info("----------Reached the Goal!----------")
                reward += self.goal_reward + 1

            if (
                goal_distance_after >= self.goal_range
                and -self.noise_tolerance <= delta_changes <= self.noise_tolerance
            ):
                reward += -0.5
        else:
            reward = -1

        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        print(object_current, reward)

        return reward, done

    # overriding method
    def _reward_function_staged(
        self, previous_environment_info, current_environment_info
    ):
        self.goal_range = 25
        self.goal_reward = 4  # goal reward minus the potential moving away negativity
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        object_previous = self._relative_position(
            previous_environment_info["poses"]["object"]
        )[:-1]
        object_current = self._relative_position(
            current_environment_info["poses"]["object"]
        )[:-1]
        logging.debug(
            f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}"
        )

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        # Staged reward system
        # ----------> S1: Hold <---------- #
        if object_current[1] <= self.bottom_line:
            reward = 1

            delta_changes = goal_distance_before - goal_distance_after

            if (
                self.total_moves < 50
            ):  # actually half of it cuz reward getting run twice per step because of render env
                # ----------> S2: Move <---------- #
                if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
                    # S2: No Move
                    reward += -0.5
                else:
                    # S2: Move
                    reward += 1
                    self.total_moves += 1
                    print("total moves: ", self.total_moves)
            else:
                # ----------> S3: Reach <---------- #
                raw_reward = delta_changes / goal_distance_before

                reward += (
                    1 if raw_reward >= 1 else -1 if raw_reward <= -1 else raw_reward
                )

                # S3: Reach Goal
                if goal_distance_after <= self.goal_range:
                    logging.info("----------Reached the Goal!----------")
                    reward += self.goal_reward + 1

                # S3: No move outside of goal range
                if (
                    goal_distance_after >= self.goal_range
                    and -self.noise_tolerance <= delta_changes <= self.noise_tolerance
                ):
                    reward += -0.5
        # S1: Drop
        else:
            reward = -1

        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        print(object_current, reward)

        return reward, done

    # overriding method touch_staged
    def _reward_function_touch_staged(
        self, previous_environment_info, current_environment_info
    ):
        self.goal_range = 25
        self.goal_reward = 4  # goal reward minus the potential moving away negativity
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        object_previous = self._relative_position(
            previous_environment_info["poses"]["object"]
        )[:-1]
        object_current = self._relative_position(
            current_environment_info["poses"]["object"]
        )[:-1]
        logging.debug(
            f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}"
        )

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        logging.debug(f"Distance to Goal: {goal_distance_after}")

        # Staged reward system
        # ----------> S1: Hold <---------- #
        if object_current[1] <= self.bottom_line:
            reward = 1

            delta_changes = goal_distance_before - goal_distance_after

            if (
                self.total_moves < 50
            ):  # actually half if it cuz reward getting run twice per step because of render env
                # ----------> S2: Move <---------- #
                if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
                    # S2: No Move
                    reward += -0.5
                else:
                    # S2: Move
                    reward += 1
                    self.total_moves += 1
                    print("total moves: ", self.total_moves)
            else:
                # ----------> S3: Reach <---------- #
                raw_reward = delta_changes / goal_distance_before

                reward += (
                    1 if raw_reward >= 1 else -1 if raw_reward <= -1 else raw_reward
                )

                # S3: Reach Goal
                if goal_distance_after <= self.goal_range:
                    logging.info("----------Reached the Goal!----------")
                    reward += 5

                # S3: No move outside of goal range
                if (
                    goal_distance_after >= self.goal_range
                    and -self.noise_tolerance <= delta_changes <= self.noise_tolerance
                ):
                    reward += -0.5
        # S1: Drop
        else:
            reward = -1

        # Touch Sensor Reward
        if self.use_touch:
            left = current_environment_info["touch"][0]
            right = current_environment_info["touch"][1]

            reward += 0.5 if left or right else 0

        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        print(object_current, reward)

        return reward, done
