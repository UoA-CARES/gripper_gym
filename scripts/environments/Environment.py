from abc import ABC, abstractmethod

import logging
import numpy as np
import random

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from Gripper import Gripper, GripperError
from configurations import EnvironmentConfig, GripperConfig

from cares_lib.vision.Camera import Camera
from cares_lib.vision.ArucoDetector import ArucoDetector

class Environment(ABC):
    def __init__(self, env_config : EnvironmentConfig, gripper_config : GripperConfig):
        self.gripper = Gripper(gripper_config)
        self.camera = Camera(env_config.camera_id, env_config.camera_matrix, env_config.camera_distortion)
        
        self.observation_type = env_config.observation_type
        self.object_type = env_config.object_type

        self.goal_selection_method = env_config.goal_selection_method
        self.noise_tolerance = env_config.noise_tolerance
        
        self.aruco_detector = ArucoDetector(marker_size=env_config.marker_size)
        self.object_marker_id = self.gripper.num_motors + 3# Num Servos + Finger Tips (2) + 1

        self.goal_state = self.get_object_state()

    def reset(self):
        try:
            self.gripper.home()
        except GripperError as error:
            # handle what to do if the gripper is unrecoverably gone wrong - i.e. save data and fail gracefully
            logging.error(error)
            exit()

        
        state = self.get_state()
        while state[-1] == -1:
            if not self.gripper.is_home():
                self.gripper.home()
            state = self.get_state()

        logging.debug(state)

        self.goal_state = self.choose_goal()

        logging.info(f"New Goal Generated: {self.goal_state}")
        return state

    def sample_action(self):
        action = []
        for i in range(0, self.gripper.num_motors):
            min_value = self.gripper.min_values[i]
            max_value = self.gripper.max_values[i]
            action.append(random.randint(min_value, max_value))
        return action
    
    def sample_action_velocity(self):
        action = []
        for i in range(0, self.gripper.num_motors):
            action.append(random.randint(self.gripper.velocity_min, self.gripper.velocity_max))
        return action

    def step(self, action): 
        # Get initial pose of the object before moving to help calculate reward after moving
        object_state_before = self.get_object_state()
        
        try:#TODO change?
            if self.observation_type == 3:
                self.gripper.move_velocity(action)
            else:
                self.gripper.move(action)
            self.gripper.step()
        except GripperError as error:
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

    def servo_state_space(self):
        # Angle Servo + X-Y-Yaw Object
        state = []
        gripper_state = self.gripper.state()
        state += gripper_state["positions"]

        object_state = self.get_object_state()
        if object_state is not None:
            position    = object_state["position"]
            orientation = object_state["orientation"]
            state.append(position[0])#X
            state.append(position[1])#Y
            state.append(orientation[2])#Yaw
        else:
            # if target is not visible then append -1 to the state (norm 0-360)
            # TODO this needs further consideration...
            state.append(-1)

        return state

    # The aruco state presumes the Aruco IDs match the servo IDs + 2 Markers for finger tips + 1 Marker for Object
    def aruco_state_space(self):
        # X-Y Servo + X-Y Finger-tips + X-Y-Yaw Object
        state = []
        
        num_markers = self.gripper.num_motors + 3 # Servos + Finger Tips (2) + Object (1)
        marker_ids = [id for id in range(1, num_markers+1)]# maker_ids match servo ids (counting from 1)
        while True:
            logging.debug(f"Attempting to Detect State")
            frame        = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix, self.camera.camera_distortion)

            # This will check that all the markers are detected correctly
            if all(ids in marker_poses for ids in marker_ids):
                break

        # Add the XY poses for each of the markers into the state
        state = [0 for _ in range(num_markers*2+1)]
        for id in marker_ids:
            marker_pose = marker_poses[id]
            position    = marker_pose["position"]
            #orientation = marker_pose["orientation"]

            i = id - 1
            state[i*2]   = position[0] # X
            state[i*2+1] = position[1] # Y
        
        # Add the additional yaw information from the object marker (adds to the end)
        state[-1:] = [marker_poses[self.object_marker_id]["orientation"][2]]#Yaw

        return state

    def servo_aruco_state_space(self):
        # Angle Servo + X-Y-Yaw Object
        state = []
        gripper_state = self.gripper.state()
        state += gripper_state["positions"]

        object_state = self.get_object_state()
        if object_state is not None:
            position    = object_state["position"]
            orientation = object_state["orientation"]
            state.append(position[0])#X
            state.append(position[1])#Y
            state.append(orientation[2])#Yaw
        else:
            # if target is not visible then append -1 to the state (norm 0-360)
            # TODO this needs further consideration...
            state.append(-1)

        # X-Y-Agnle Servo + X-Y Finger Tips + X-Y-Yaw Object
        state_size = self.gripper.num_motors * 3 + 7 # Num Servos * 3 + Finger Tips * 2 (4) + Object (3)

        state = [0 for _ in range(state_size)]
        servo_state_space = self.servo_state_space()
        aruco_state_space = self.aruco_state_space()

        # Add servo state space X-Y-Angle
        for i in range(0, self.gripper.num_motors):
            s_i = i * 3
            a_i = i * 2
            state[s_i]   = aruco_state_space[a_i]   # X
            state[s_i+1] = aruco_state_space[a_i+1] # Y
            state[s_i+2] = servo_state_space[i]     # Angle

        # Add Finger Tips
        m_i = self.gripper.num_motors * 3
        a_i = self.gripper.num_motors * 2 
        state[m_i:m_i+5] = aruco_state_space[a_i:a_i+5]# (4+1) one extra to get full range

        # Add Object Target
        state[-3:] = aruco_state_space[-3:]

        return state
    
    def servo_velocity_state_space(self):
        # Angle Servo + X-Y-Yaw Object
        state = []
        self.object_marker_id = 1
        gripper_state = self.gripper.state()
        state += gripper_state["positions"]
        state += gripper_state["velocities"]
        state += gripper_state["loads"]

        object_state = self.get_object_state()
        if object_state is not None:
            position    = object_state["position"]
            orientation = object_state["orientation"]
            state.append(position[0])#X
            state.append(position[1])#Y
            state.append(orientation[2])#Yaw
        else:
            # if target is not visible then append -1 to the state (norm 0-360)
            # TODO this needs further consideration...
            state.append(-1)

        return state

        
    #TODO implement function
    def image_state_space(self):
        # Note should store the stacked frame somewhere...
        raise NotImplementedError("Requires implementation")

    def get_state(self):
        if self.observation_type == 0:# TODO Turn into enum
            return self.servo_state_space()
        elif self.observation_type == 1:
            return self.aruco_state_space()
        elif self.observation_type == 2:
            return self.servo_aruco_state_space()
        elif self.observation_type == 3:
            return self.servo_velocity_state_space()
        
        raise ValueError(f"Observation Type unknown: {self.observation_type}")

    def get_aruco_target_pose(self, blindable=False, detection_attempts=4):
        attempt = 0
        while not blindable or attempt < detection_attempts: 
            attempt += 1
            msg = f"{attempt}/{detection_attempts}" if blindable else f"{attempt}"
            logging.debug(f"Attempting to detect aruco target: {msg}")
            logging.info(f"Attempting to detect aruco target: {msg}")

            frame = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix, self.camera.camera_distortion)
            if self.object_marker_id in marker_poses:
                return marker_poses[self.object_marker_id]
        return None

    def get_object_state(self):
        if self.object_type == 0:
            return self.get_aruco_target_pose(blindable=False)
        elif self.object_type == 1:
            return self.get_aruco_target_pose(blindable=True)

        raise ValueError(f"Unknown object type: {self.object_type}")

    def denormalize(self, action_norm):
        # return action in gripper range [-min, +max] for each servo
        action_gripper = [0 for _ in range(0, len(action_norm))]
        min_value_in = -1
        max_value_in = 1
        for i in range(0, self.gripper.num_motors):
            servo_min_value = self.gripper.min_values[i]
            servo_max_value = self.gripper.max_values[i]
            action_gripper[i] = int((action_norm[i] - min_value_in) * (servo_max_value - servo_min_value) / ( max_value_in - min_value_in) + servo_min_value)
        return action_gripper

    def normalize(self, action_gripper):
        # return action in algorithm range [-1, +1]
        max_range_value = 1
        min_range_value = -1
        action_norm = [0 for _ in range(0, len(action_gripper))]
        for i in range(0, self.gripper.num_motors):
            servo_min_value = self.gripper.min_values[i]
            servo_max_value = self.gripper.max_values[i]
            action_norm[i]  = (action_gripper[i] - servo_min_value) * (max_range_value - min_range_value) / (servo_max_value - servo_min_value) + min_range_value
        return action_norm


    @abstractmethod
    def choose_goal(self):
        pass

    @abstractmethod
    def reward_function(self, target, start_target_pose, final_target_pose):
        pass