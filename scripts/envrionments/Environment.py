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

##### Find object type pose functions
def get_aruco_target_pose(marker_id, camera, aruco_detector, blindable=False, detection_attempts=4):
    attempt = 0
    while not blindable or attempt < detection_attempts:
        msg = f"{attempt}/{detection_attempts}" if blindable else f"{attempt}"
        logging.debug(f"Attempting to detect aruco target: {msg}")
        frame = camera.get_frame()
        marker_poses = aruco_detector.get_marker_poses(frame, camera.camera_matrix, camera.camera_distortion)
        if marker_id in marker_poses:
            return marker_poses[marker_id]
        attempt += 1
    return None

def get_servo_target_pose(gripper, marker_id, camera, aruco_detector, blindable=False, detection_attempts=4):
    object_pose = get_aruco_target_pose(marker_id, camera, aruco_detector, blindable, detection_attempts)
    object_pose["orientation"][2] = gripper.target_servo.current_position()
    return object_pose
#####

class Environment(ABC):
    def __init__(self, env_config : EnvironmentConfig, gripper_config : GripperConfig):
        self.gripper = ghlp.create_gripper(gripper_config)
        self.camera = Camera(env_config.camera_id, env_config.camera_matrix, env_config.camera_distortion)
        
        self.observation_type = env_config.observation_type
        self.object_type = env_config.object_type

        self.goal_selection_method = env_config.goal_selection_method
        self.noise_tolerance = env_config.noise_tolerance
        
        self.aruco_detector = ArucoDetector(marker_size=env_config.marker_size)
        self.object_marker_id = self.gripper.num_motors + 1

        self.goal_state = self.get_object_state()

    def reset(self):
        try:
            self.gripper.home()
        except DynamixelServoError as error:
            # handle what to do if the gripper is unrecoverably gone wrong - i.e. save data and fail gracefully
            logging.error(error)
            exit()

        state = self.get_state()
        logging.info(state)

        self.goal_state = self.choose_goal()

        logging.info(f"New Goal Generated: {self.goal_state}")
        return state

    def step(self, action):
        
        # Get initial pose of the marker before moving to help calculate reward after moving
        object_state_before = self.get_object_state()
        
        try:
            self.gripper.move(action)
        except DynamixelServoError as error:
            # handle what to do if the gripper is unrecoverably gone wrong - i.e. save data and fail gracefully
            logging.error(error)
            exit()

        state = self.get_state()
        logging.debug(f"New State: {state}")

        object_state_after = self.get_object_state()
        logging.debug(f"New Object State: {object_state_after}")
        
        reward, done = self.reward_function(self.goal_state, object_state_before, object_state_after)

        truncated = False #never truncate the episode but here for completion sake
        return state, reward, done, truncated

    def gripper_state(self):
        state = self.gripper.current_positions()
        object_state = self.get_object_state()
        
        # if target is not visible then append -1 to the state (norm 0-360)
        if object_state is not None:
            position    = object_state["position"]
            orientation = object_state["orientation"]
            state.append(position[0])#X
            state.append(position[1])#Y
            state.append(orientation[2])#Yaw
        else:
            state.append(-1)

        return state

    def marker_state(self):
        # This all presumes the Aruco IDs match the servo IDs and the object is number of servos plus one

        # X-Y positions of servos + X-Y-Yaw of target
        state = []
        
        num_markers = self.gripper.num_motors + 1 # Plus the target object
        marker_ids = [id for id in range(1, num_markers+1)]# maker_ids match servo ids (counting from 1)
        while True:
            logging.debug(f"Attempting to Detect State")
            frame        = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix, self.camera.camera_distortion)

            # This will check that all the markers are detected correctly
            if len(marker_poses) >= len(marker_ids) and all(ids in marker_poses for ids in marker_ids):
                break

        # Add the XY poses for each of the markers into the state
        state = [0 for _ in range(num_markers*2+1)]
        
        for id in marker_ids:
            marker_pose = marker_poses[id]
            position    = marker_pose["position"]
            orientation = marker_pose["orientation"]

            i = id - 1
            state[i*2]   = position[0]#X
            state[i*2+1] = position[1]#Y
        
        # Add the additional yaw information from the object marker
        i = self.object_marker_id - 1
        state[i*2+2] = marker_poses[self.object_marker_id]["orientation"][2]#Yaw
        return state

    def get_state(self):
        if self.observation_type == 0:# TODO Turn into enum
            return self.gripper_state()
        elif self.observation_type == 1:
            return self.marker_state()
        
        raise ValueError(f"Observation Type unkown: {self.observation_type}")

    def get_object_state(self):
        if self.object_type == 0:
            return get_aruco_target_pose(self.object_marker_id, self.camera, self.aruco_detector, False)
        elif self.object_type == 1:
            return get_aruco_target_pose(self.object_marker_id, self.camera, self.aruco_detector, True, 4)
        elif self.object_type == 2:
            return get_servo_target_pose(self.gripper)

        raise ValueError(f"Unkown object type: {self.object_type}")

    @abstractmethod
    def choose_goal(self):
        pass

    @abstractmethod
    def reward_function(self, target, start_target_pose, final_target_pose):
        pass