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
    return 90

def fixed_goals(object_current_pose, noise_tolerance):
    current_yaw = object_current_pose# Yaw.

    target_angle = fixed_goal()
    while abs(current_yaw - target_angle) < noise_tolerance:
        target_angle = fixed_goal()
    return target_angle

def relative_goal(object_current_pose):
    current_yaw = object_current_pose['orientation'][2]# Yaw.
    return current_yaw + 90 #TODO redo this

# TODO turn the hard coded type ints into enums
class RotationEnvironment(Environment):
    def __init__(self, env_config : EnvironmentConfig, gripper_config : GripperConfig):
        super().__init__(env_config, gripper_config)

    # overriding method
    def choose_goal(self):
        object_state = self.actual_object_state() 
        if self.goal_selection_method == 0:# TODO Turn into enum
            return fixed_goals(object_state, self.noise_tolerance)
        elif self.goal_selection_method == 1:
            return relative_goal(object_state)
        
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

        yaw_before = goal_before
        yaw_after  = goal_after

        goal_difference = self.rotation_min_difference(target_goal, yaw_after)
        delta_changes   = self.rotation_min_difference(target_goal, yaw_before) - self.rotation_min_difference(target_goal, yaw_after)
        
        logging.info(f"Yaw = {yaw_after}")

        no_action_tolerance = 3 # TODO should this not be self.noise_tolerance
        if -no_action_tolerance <= delta_changes <= no_action_tolerance:
            reward = -1
        else:
            reward = delta_changes/self.rotation_min_difference(target_goal, yaw_before)

        if goal_difference <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward += 1
            done = True

        return reward, done
