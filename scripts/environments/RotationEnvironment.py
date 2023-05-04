from environments.Environment import Environment

import logging
import numpy as np

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from configurations import EnvironmentConfig, GripperConfig

def fixed_goal():
    # target_angle = np.random.randint(1, 5)
    # if target_angle == 1:
    #     return 90
    # elif target_angle == 2:
    #     return 180
    # elif target_angle == 3:
    #     return 270
    # elif target_angle == 4:
    #     return 0
    return 270
    raise ValueError(f"Target angle unknown: {target_angle}")

def fixed_goals(object_current_pose, noise_tolerance):
    if type(object_current_pose) == dict:
        current_yaw = object_current_pose['orientation'][2]# Yaw.
    else:
        current_yaw = object_current_pose#['orientation'][2]# Yaw.

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
            if self.gripper.actuated_target:
                object_state = self.gripper.current_object_position()#self.get_object_state()
            else:
                object_state = self.get_object_state() 


            return fixed_goals(object_state, self.noise_tolerance)
        elif self.goal_selection_method == 1:
            if self.gripper.actuated_target:
                return relative_goal(self.gripper.current_object_position())#self.get_object_state()
            else:
                return relative_goal(self.get_object_state())

            return relative_goal(self.gripper.current_object_position())
        
        raise ValueError(f"Goal selection method unknown: {self.goal_selection_method}")

    def min_difference(self, a, b):
        return min(abs(a - b), (360+min(a, b) - max(a, b)))
        # return min((a - b), (360+min(a, b) - max(a, b)))
    
    # overriding method 
    def reward_function(self, target_goal, goal_before, goal_after):
        if goal_before is None: 
            logging.debug("Start Marker Pose is None")
            return 0, True

        if goal_after is None:
            logging.debug("Final Marker Pose is None")
            return 0, True
        
        done = False


        if type(goal_before) == dict:
            yaw_before = goal_before["orientation"][2]
            yaw_after  = goal_after["orientation"][2]
        else:
            yaw_before = goal_before
            yaw_after  = goal_after

        # yaw_before = goal_before#["orientation"][2]
        # yaw_after  = goal_after#["orientation"][2]

        goal_difference = self.min_difference(target_goal, yaw_after)
        # To a goal position
        delta_changes   = self.min_difference(target_goal, yaw_before) - self.min_difference(target_goal, yaw_after)

        # Reward any change
        # delta_changes = self.min_difference(yaw_before - yaw_after)

        # Reward only changes in the right direction
        # delta_changes = yaw_after - yaw_before

        logging.info(f"Yaw = {yaw_after}")

        if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
            reward = -1
        else:
            reward = delta_changes/self.min_difference(target_goal, yaw_before)

            # reward = reward if reward > 0 else -2

        if goal_difference <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward += 1
            done = True

        return reward, done
