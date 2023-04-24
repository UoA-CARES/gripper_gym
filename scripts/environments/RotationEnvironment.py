from environments.Environment import Environment

import logging
import numpy as np

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from configurations import EnvironmentConfig, GripperConfig

def fixed_goal():
    target_angle = np.random.randint(1, 5)
    if target_angle == 1:
        return 90
    elif target_angle == 2:
        return 180
    elif target_angle == 3:
        return 270
    elif target_angle == 4:
        return 0
    raise ValueError(f"Target angle unknown: {target_angle}")

def fixed_goals(object_current_pose, noise_tolerance):
    current_yaw = object_current_pose['orientation'][2]# Yaw
    target_angle = fixed_goal()
    while abs(current_yaw - target_angle) < noise_tolerance:
        target_angle = fixed_goal()
    return target_angle

def relative_goal(current_target):
    return current_target + 90 #TODO redo this
#####

# TODO turn the hard coded type ints into enums
class RotationEnvironment(Environment):
    def __init__(self, env_config : EnvironmentConfig, gripper_config : GripperConfig):
        super().__init__(env_config, gripper_config)

    # overriding method
    def choose_goal(self):
        if self.goal_selection_method == 0:# TODO Turn into enum
            object_state = self.get_object_state()
            while object_state == None:
                if not self.gripper.is_home():
                    self.gripper.home()
                object_state = self.get_object_state()

            return fixed_goals(object_state, self.noise_tolerance)
        elif self.goal_selection_method == 1:
            return relative_goal(self.get_object_state())
        
        raise ValueError(f"Goal selection method unknown: {self.goal_selection_method}")

    # overriding method 
    def reward_function(self, target_goal, goal_before, goal_after):
        if goal_before is None: 
            logging.debug("Start Marker Pose is None")
            return 0, True

        if goal_after is None:
            logging.debug("Final Marker Pose is None")
            return 0, True
        
        done = False

        yaw_before = goal_before["orientation"][2]
        yaw_after  = goal_after["orientation"][2]

        goal_difference = np.abs(target_goal - yaw_after)
        delta_changes   = np.abs(target_goal - yaw_before) - np.abs(target_goal - yaw_after)

        logging.info(f"Yaw = {yaw_after}")

        if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
            reward = -10
        else:
            reward = delta_changes/abs(target_goal - yaw_before)
            reward = reward if reward > 0 else -10

        if goal_difference <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            done = True

        return reward, done
