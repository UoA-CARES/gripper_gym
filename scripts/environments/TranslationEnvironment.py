from environments.Environment import Environment

import logging
import numpy as np

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from configurations import EnvironmentConfig, GripperConfig

from cares_lib.vision.Camera import Camera
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.dynamixel.Servo import DynamixelServoError

##### Set goal functions

#####

# TODO turn the hard coded type ints into enums
class TranslationEnvironment(Environment):
    def __init__(self, env_config : EnvironmentConfig, gripper_config : GripperConfig):
        super().__init__(env_config, gripper_config)

    # overriding method
    def choose_goal(self):
        position = self.get_object_state()['position']
        target_index = np.random.randint(0, 1)
        if target_index == 1:
            position[0] = np.random.randint(340, 430)
            position[1] = np.random.randint(310, 380)
        else:
            position[0] = np.random.randint(740, 840)
            position[1] = np.random.randint(310, 380)
        # # raise ValueError(f"Goal selection method unkown: {self.goal_selection_method}")
        # raise NotImplementedError("Reward not implement")
        return position

    # overriding method 
    def reward_function(self, target_goal, goal_before, goal_after):
        print(target_goal)
        print(goal_before)
        print(goal_after)
        raise NotImplementedError("Reward not implement")
    