import logging
import math
from random import randrange

from cares_lib.dynamixel.gripper_configuration import GripperConfig
from configurations import GripperEnvironmentConfig, ObjectConfig
from environments.two_finger.two_finger import TwoFingerTask


class TwoFingerTranslation(TwoFingerTask):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
        object_config: ObjectConfig,
    ):
        self.noise_tolerance = 15

        # These bounds are respective to the reference marker in Environment
        self.goal_min = [-30.0, 60.0]
        self.goal_max = [120.0, 110.0]
        logging.info(f"Goal Max: {self.goal_max}")
        logging.info(f"Goal Min: {self.goal_min}")

        super().__init__(env_config, gripper_config, object_config)

    # overriding method
    def _choose_goal(self):
        x1, y1 = self.goal_min
        x2, y2 = self.goal_max

        goal_x = randrange(x1, x2)
        goal_y = randrange(y1, y2)

        logging.info(f"New Goal: {goal_x}, {goal_y}")
        return [goal_x, goal_y]

    # overriding method
    def _environment_info_to_state(self, environment_info):
        state = []

        # Servo Angles - Steps
        state += environment_info["gripper"]["positions"]

        # Servo Velocities - Steps per second
        if self.action_type == "velocity":
            state += environment_info["gripper"]["velocities"]

        # Servo + Two Finger Tips - X Y mm
        for i in range(1, self.gripper.num_motors + 3):
            servo_position = environment_info["poses"]["gripper"][i]
            state += self._pose_to_state(servo_position)

        # Object - X Y mm
        state += self._pose_to_state(environment_info["poses"]["object"])

        # Goal State - X Y mm
        state += self.goal

        return state

    # overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        object_previous = previous_environment_info["poses"]["object"]["position"][0:2]
        object_current = current_environment_info["poses"]["object"]["position"][0:2]

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        goal_progress = goal_distance_before - goal_distance_after

        # The following step might improve the performance.

        # goal_before_array = goal_before[0:2]
        # delta_changes   = np.linalg.norm(target_goal - goal_before_array) - np.linalg.norm(target_goal - goal_after_array)
        # if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
        #     reward = -10
        # else:
        #     reward = -goal_difference
        #     #reward = delta_changes / (np.abs(yaw_before - target_goal))
        #     #reward = reward if reward > 0 else 0

        # For Translation. noise_tolerance is 15, it would affect the performance to some extent.
        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            done = True
            reward = 500
        else:
            reward += goal_progress

        logging.info(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        return reward, done


class TwoFingerTranslationFlat(TwoFingerTranslation):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
        object_config: ObjectConfig,
    ):
        super().__init__(env_config, gripper_config, object_config)

    # overriding method
    def _reset(self):
        self.gripper.wiggle_home()


class TwoFingerTranslationSuspended(TwoFingerTranslation):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
        object_config: ObjectConfig,
    ):
        super().__init__(env_config, gripper_config, object_config)

        # TODO add instatiation of the elevator servo etc here

    # overriding method
    def _reset(self):
        self.gripper.wiggle_home()

        # Execute code to reset the elevator etc
