from environments.Environment import Environment

import logging
import numpy as np

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from configurations import EnvironmentConfig, GripperConfig, ObjectConfig

from cares_lib.vision.Camera import Camera
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.dynamixel.Servo import DynamixelServoError

##### Set goal functions

#####

# TODO turn the hard coded type ints into enums
class TranslationEnvironment(Environment):
    def __init__(self, env_config : EnvironmentConfig, gripper_config : GripperConfig, object_config: ObjectConfig):
        super().__init__(env_config, gripper_config, object_config)

    # overriding method
    def choose_goal(self):
        # raise ValueError(f"Goal selection method unkown: {self.goal_selection_method}")
        raise NotImplementedError("Reward not implement")

    # overriding method 
    def reward_function(self, target_goal, goal_before, goal_after):
        raise NotImplementedError("Reward not implement")
    