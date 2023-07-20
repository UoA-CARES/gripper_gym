from abc import ABC, abstractmethod

import logging
import random
from functools import wraps
from scipy.stats import trim_mean
from pathlib import Path
from enum import Enum
import numpy as np

file_path = Path(__file__).parent.resolve()

from configurations import EnvironmentConfig, GripperConfig, ObjectConfig
from Gripper import Gripper
from Objects import MagnetObject, ServoObject

from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.vision.Camera import Camera

def exception_handler(error_message):
    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            try:
                return function(self, *args, **kwargs)
            except EnvironmentError as error:
                logging.error(f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}")
                raise EnvironmentError(error.gripper, f"Environment for Gripper#{error.gripper.gripper_id}: {error_message}") from error
        return wrapper
    return decorator

class EnvironmentError(IOError):
    def __init__(self, gripper, message):
        self.gripper = gripper
        super().__init__(message)

class OBSERVATION_TYPE(Enum):
    SERVO = 0
    ARUCO = 1
    SERVO_ARUCO = 2
    IMAGE = 3

VALVE_SERVO_ID = 10

class Environment(ABC):
    def __init__(self, env_config: EnvironmentConfig, gripper_config: GripperConfig, object_config: ObjectConfig):
        
        self.gripper = Gripper(gripper_config)
        self.camera  = Camera(env_config.camera_id, env_config.camera_matrix, env_config.camera_distortion)

        self.observation_type = env_config.observation_type
        self.action_type = env_config.action_type

        self.blindable = env_config.blindable

        self.goal_selection_method = env_config.goal_selection_method
        self.noise_tolerance = env_config.noise_tolerance

        self.gripper.home()
        self.aruco_detector = ArucoDetector(marker_size=env_config.marker_size)

        self.object_marker_id = object_config.object_marker_id
        self.object_observation_mode = object_config.object_observation_mode

        aruco_yaw = None
        if self.object_observation_mode == "observed":
            aruco_yaws = []
            for i in range(0, 10):
                aruco_yaws.append(self.observed_object_state(marker_only=True)[2])
            aruco_yaw = trim_mean(aruco_yaws, 0.1)

        self.object_type = object_config.object_type
        if self.object_type == "magnet":
            self.target = MagnetObject(object_config, aruco_yaw)
        elif self.object_type == "servo":
            self.target = ServoObject(object_config, VALVE_SERVO_ID)
        else:
            raise ValueError("Object Type unknown")

        self.goal_state = self.actual_object_state()

    @exception_handler("Environment failed to reset")
    def reset(self):
        
        self.gripper.wiggle_home()  
        self.target.reset_target_servo() # only reset if using new servos
        state = self.get_state()

        logging.debug(state)

        # choose goal will crash if not home
        self.goal_state = self.choose_goal()

        logging.info(f"New Goal Generated: {self.goal_state}")
        return state
        
    def sample_action(self):
        if self.action_type == "velocity":
            return self.sample_action_velocity()
        return self.sample_action_position()

    def sample_action_position(self):
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

    @exception_handler("Failed to step")
    def step(self, action):
        object_state_before = self.actual_object_state()
        
        if self.action_type == "velocity":
            self.gripper.move_velocity(action, False)
        else:
            self.gripper.move(action)
        
        state = self.get_state()
        logging.debug(f"New State: {state}")

        object_state_after = self.actual_object_state()

        logging.debug(f"New Object State: {object_state_after}")

        reward, done = self.reward_function(self.goal_state, object_state_before, object_state_after)

        truncated = False
        return state, reward, done, truncated
    
    @exception_handler("Failed to step gripper")
    def step_gripper(self):
        self.gripper.step()

    @exception_handler("Failed to get servo states")
    def servo_state_space(self):
        # Angle Servo + X-Y-Yaw Object + Goal
        state = []
        gripper_state = self.gripper.state()
        state += gripper_state["positions"]

        if self.action_type == "velocity":
            state += gripper_state["velocities"]
            state += gripper_state["loads"]

        if self.object_observation_mode == "observed":
            state += self.observed_object_state()
        elif self.object_observation_mode == "actual":
            state += self.actual_object_state(yaw_only=False)

        state.append(self.goal_state)

        return state 

    # The aruco state presumes the Aruco IDs match the servo IDs + a Marker for each finger tip + 1 Marker for Object
    @exception_handler("Failed to get aruco states")
    def aruco_state_space(self):
        # X-Y Servo + X-Y Finger-tips + X-Y-Yaw Object + Goal
        state = []

        num_markers = self.gripper.num_motors + 3  # Servos + Finger Tips (2) + Object (1)
        marker_ids = [id for id in range(1, num_markers + 1)]  # maker_ids match servo ids (counting from 1)
        while True:
            logging.debug(f"Attempting to Detect State")
            frame = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix,
                                                                self.camera.camera_distortion, display=False)

            # This will check that all the markers are detected correctly
            if all(ids in marker_poses for ids in marker_ids):
                break

        # order the markers by ID
        marker_poses = dict(sorted(marker_poses.items()))

        # Add the XY poses for each of the markers into the state
        for _, marker_pose in marker_poses.items():
            position = marker_pose["position"]
            state.append(position[0])  # X
            state.append(position[1])  # Y

        # Add the additional yaw information from the object marker (adds to the end)
        state[-1:] = [marker_poses[self.object_marker_id]["orientation"][2]]  # Yaw

        state.append(self.goal_state)

        return state


    @exception_handler("Failed to get servo and aruco states")
    def servo_aruco_state_space(self):
        # Servo (Position/Velocity/Load) + Servo XY + Target XY-Yaw + Goal
        state = []
        servo_state_space = self.servo_state_space()[:-4]#remove the redundent target XY-Yaw + goal from the end
        aruco_state_space = self.aruco_state_space()[:-1]#remove the goal from the end

        state += servo_state_space
        state += aruco_state_space

        state.append(self.goal_state)

        return state

    # TODO implement function
    def image_state_space(self):
        # Note should store the stacked frame somewhere...
        raise NotImplementedError("Requires implementation")

    @exception_handler("Failed to get state")
    def get_state(self): 
        if self.observation_type == OBSERVATION_TYPE.SERVO.value:
            return self.servo_state_space()
        elif self.observation_type == OBSERVATION_TYPE.ARUCO.value:
            return self.aruco_state_space()
        elif self.observation_type == OBSERVATION_TYPE.SERVO_ARUCO.value:
            return self.servo_aruco_state_space()
        elif self.observation_type == OBSERVATION_TYPE.IMAGE.value:
            return self.image_state_space()
        
        raise ValueError(f"Observation Type unknown: {self.observation_type}") 

    def get_aruco_object_pose(self, blindable=False, detection_attempts=4):
        attempt = 0
        while not blindable or attempt < detection_attempts: 
            attempt += 1
            msg = f"{attempt}/{detection_attempts}" if blindable else f"{attempt}"
            logging.debug(f"Attempting to detect aruco target: {self.object_marker_id}")

            frame = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix,
                                                                self.camera.camera_distortion, display=False)
            if self.object_marker_id in marker_poses:
                return marker_poses[self.object_marker_id]
        return None
    
    def observed_object_state(self, marker_only=False):
        object_state = self.get_aruco_object_pose(blindable=self.blindable, detection_attempts=5)
        if object_state is not None:
            state = []
            position    = object_state["position"]
            orientation = object_state["orientation"]
            state.append(position[0])  # X
            state.append(position[1])  # Y
            state.append(orientation[2])  # Yaw

            if not marker_only:
                angle_offsets = [45, 135, 225, 315]
                state+=self.get_object_ends_pose(orientation[2], angle_offsets, center_x=position[0], center_y=position[1])
            return state
        return [0]*11    
        
    @exception_handler("Failed to get object states")
    def actual_object_state(self, yaw_only=True):
        yaw = self.target.get_yaw()
        if yaw_only:
            return yaw
        else:
            state = [yaw]
            angle_offsets = [0, 90, 180, 270]
            state += self.get_object_ends_pose(yaw, angle_offsets)
            return state
            
    def get_object_ends_pose(self, center_yaw, angle_offsets, center_x = 0, center_y = 0):
        object_ends = [0]*8
        ends_distance =  5.2

        for i in range(4):
            angle = center_yaw + angle_offsets[i]
            object_ends[i*2] = center_x + np.sin(np.deg2rad(angle)) * ends_distance
            object_ends[i*2+1] = center_y + np.cos(np.deg2rad(angle)) * ends_distance

        return object_ends

    def denormalize(self, action_norm):
        # return action in gripper range [-min, +max] for each servo
        action_gripper = [0 for _ in range(0, len(action_norm))]
        min_value_in = -1
        max_value_in = 1
        for i in range(0, self.gripper.num_motors):
            if self.action_type == "velocity":
                servo_min_value = self.gripper.velocity_min
                servo_max_value = self.gripper.velocity_max
            else:
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
            if self.action_type == "velocity":
                servo_min_value = self.gripper.velocity_min
                servo_max_value = self.gripper.velocity_max
            else:
                servo_min_value = self.gripper.min_values[i]
                servo_max_value = self.gripper.max_values[i]
            action_norm[i]  = (action_gripper[i] - servo_min_value) * (max_range_value - min_range_value) / (servo_max_value - servo_min_value) + min_range_value
        return action_norm

    def ep_final_distance(self):
        return self.rotation_min_difference(self.goal_state, self.actual_object_state())
    
    def rotation_min_difference(self, a, b):
        return min(abs(a - b), (360+min(a, b) - max(a, b)))
    
    @abstractmethod
    def choose_goal(self):
        pass

    @abstractmethod
    def reward_function(self, target, start_target_pose, final_target_pose):
        pass
