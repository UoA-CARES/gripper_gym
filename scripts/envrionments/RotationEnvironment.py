from abc import ABC
from envrionments.Environment import Environment

import logging
import numpy as np

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from pydantic import BaseModel
from typing import List, Optional

from configurations import EnvironmentConfig, GripperConfig
import grippers.gripper_helper as ghlp

from cares_lib.vision.Camera import Camera
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.dynamixel.Servo import DynamixelServoError

def fixed_goals():
    target_angle = np.random.randint(1,5)
    if target_angle == 1:
        return 90
    elif target_angle == 2:
        return 180
    elif target_angle == 3:
        return 270
    elif target_angle == 4:
        return 0
    return 180 # Default to 180 but shouldn't get to this return

def relative_goal(current_target):
    return current_target + 90 #TODO redo this

def get_target_pose(self):
    detect_attempts = 4
    for i in range(0, detect_attempts):
        logging.debug(f"Attempting to detect marker attempt {i}/{detect_attempts}")
        frame = self.camera.get_frame()
        marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix, self.camera.camera_distortion)
        if self.marker_id in marker_poses:
            return marker_poses[self.marker_id]
    return None

class RotationEnvironment(Environment):
    def __init__(self, env_config : EnvironmentConfig, gripper_config : GripperConfig):
        
        self.aruco_detector = ArucoDetector(marker_size=env_config.marker_size)
        self.marker_id = env_config.marker_id
        
        self.noise_tolerance = 5 # TODO move to config
        self.observation_type = 0 # TODO move to config
        self.goal_selection_method = 0 # TODO move to config

        super().__init__(env_config, gripper_config)

    def gripper_state(self):
        state = self.gripper.current_positions()
        object_state = self.get_object_pose()
        
        # if target is not visible then append -1 to the state (norm 0-360)
        if object_state is not None:
            state.append(object_state)
        else:
            state.append(-1)

        return state

    def marker_state(self):    
        state_size = self.gripper.num_motors + 1
        state = [0 for _ in range(state_size)]
        marker_ids = [id for id in range(state_size)]
        while True:
            logging.debug(f"Attempting to Detect Markers")
            frame        = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix, self.camera.camera_distortion)

            # This check that all the markers are detected correctly
            if len(marker_poses) == len(marker_ids) and all(ids in marker_poses for ids in marker_ids):
                break

        for marker_id in marker_ids:
            state[marker_id] = marker_poses[marker_id][1][2] # TODO untangle the messy orientation in ArucoDetector
        
        return state

    # overriding method
    def get_state(self):
        if self.observation_type == 0:
            return self.gripper_state()
        elif self.observation_type == 1:
            return self.marker_state()
        
        raise ValueError(f"Observation Type unkown: {self.observation_type}")

    # overriding method 
    def choose_goal(self):
        if self.target_selection_method == 0:
            return fixed_goals()
        elif self.target_selection_method == 1:
            return relative_goal(self.get_object_pose())
        
        raise ValueError(f"Target selection method unkown: {self.target_selection_method}")

    # overriding method 
    def reward_function(self, target_goal, goal_before, goal_after):
        if goal_before is None: 
            logging.debug("Start Marker Pose is None")
            return 0, True

        if goal_after is None:
            logging.debug("Final Marker Pose is None")
            return 0, True
        
        done = False

        goal_difference = np.abs(target_goal - goal_after)
        delta_changes   = np.abs(target_goal - goal_before) - np.abs(target_goal - goal_after)

        reward = 0
        if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
            reward = 0
        else:
            reward = delta_changes

        if goal_difference <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = reward + 100
            done = True
        
        return reward, done

    # overriding method
    def get_object_pose(self):
        detect_attempts = 4
        for i in range(0, detect_attempts):
            logging.debug(f"Attempting to detect marker attempt {i}/{detect_attempts}")
            frame = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix, self.camera.camera_distortion)
            if self.marker_id in marker_poses:
                return marker_poses[self.marker_id]
        return None