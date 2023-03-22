from abc import ABC, abstractmethod

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
    
class Environment(ABC):
    def __init__(self, env_config : EnvironmentConfig, gripper_config : GripperConfig):
        self.gripper = ghlp.create_gripper(gripper_config)
        self.camera = Camera(env_config.camera_id, env_config.camera_matrix, env_config.camera_distortion)
        
        self.goal_pose = None

    def reset(self):
        try:
            self.gripper.home()
        except DynamixelServoError as error:
            # handle what to do if the gripper is unrecoverably gone wrong - i.e. save data and fail gracefully
            logging.error(error)
            exit()

        state = self.get_sate()

        self.goal_pose = self.choose_goal()

        logging.info(f"New Goal Generated: {self.goal_pose}")
        return state

    def step(self, action):
        
        # Get initial pose of the marker before moving to help calculate reward after moving
        object_pose_before = self.get_object_pose()
        
        try:
            self.gripper.move(action)
        except DynamixelServoError as error:
            # handle what to do if the gripper is unrecoverably gone wrong - i.e. save data and fail gracefully
            logging.error(error)
            exit()

        state = self.get_state()

        object_pose_after = self.get_object_pose()
        
        reward, done = self.reward_function(self.goal_pose, object_pose_before, object_pose_after)

        truncated = False #never truncate the episode but here for completion sake
        return state, reward, done, truncated

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def choose_goal(self):
        pass

    @abstractmethod
    def reward_function(self, target, start_target_pose, final_target_pose):
        pass

    @abstractmethod
    def get_object_pose(self):
        pass