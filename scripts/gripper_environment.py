import logging
import numpy as np

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from grippers.ArduinoGripper import ArduinoGripper
from grippers.U2D2Gripper import U2D2Gripper

from cares_lib.vision.Camera import Camera
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.dynamixel.Servo import DynamixelServoError

class GripperEnvironment():
    def __init__(self,
                 gripper_type,
                 marker_id=0,
                 marker_size=18,
                 camera_matrix_path = f"{file_path}/config/camera_matrix.txt",
                 camera_distortion_path = f"{file_path}/config/camera_distortion.txt"):
        
        if gripper_type == 0:# U2D2
            self.gripper = U2D2Gripper()
        elif gripper_type == 1:# Arduino
            self.gripper = ArduinoGripper()

        self.camera = Camera(0, camera_matrix_path, camera_distortion_path)
        
        self.aruco_detector = ArucoDetector(marker_size=marker_size)
        self.target_angle = self.choose_target_angle()

        self.marker_id = marker_id

    def reset(self):
        try:
            state = self.gripper.home()
        except DynamixelServoError as error:
            # handle what to do if the gripper is unrecoverably gone wrong - i.e. save data and fail gracefully
            logging.error(error)
            exit()

        marker_pose = self.find_marker_pose(marker_id=self.marker_id)

        if marker_pose is None:
            marker_yaw = -1# TODO replace with a raise exception?
        else:
            marker_yaw = marker_pose[1][2]

        state.append(marker_yaw)

        self.target_angle = self.choose_target_angle()

        return state

    def choose_target_angle(self):
        # David's suggestion - choose 1 of 4 angles to make training easier
        target_angle = np.random.randint(1,5)
        if target_angle == 1:
            return 90
        elif target_angle == 2:
            return 180
        elif target_angle == 3:
            return 270
        elif target_angle == 4:
            return 0
        return -1 # Should not return -1

    def reward_function(self, target_angle, start_marker_pose, final_marker_pose):
        if start_marker_pose is None: 
            logging.debug("Start Marker Pose is None")
            return 0, True

        if final_marker_pose is None:
            logging.debug("Final Marker Pose is None")
            return 0, True
        
        done = False
    
        valve_angle_before = start_marker_pose[1][2]
        valve_angle_after  = final_marker_pose[1][2]

        angle_difference = np.abs(target_angle - valve_angle_after)
        delta_changes    = np.abs(target_angle - valve_angle_before) - np.abs(target_angle - valve_angle_after)

        reward = 0
        # TODO paramatise the noise tolerance parameters
        noise_tolerance = 10
        if -noise_tolerance <= delta_changes <= noise_tolerance:
            reward = 0
        else:
            reward = delta_changes

        if angle_difference <= noise_tolerance:
            reward = reward + 100
            logging.debug("Reached the Goal Angle!")
            done = True
        
        return reward, done

    def find_marker_pose(self, marker_id):
        detect_attempts = 4
        for i in range(0, detect_attempts):
            logging.debug(f"Attempting to detect marker attempt {i}/{detect_attempts}")
            frame = self.camera.get_frame()
            marker_poses = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix, self.camera.camera_distortion)
            if marker_id in marker_poses:
                return marker_poses[marker_id]
        return None

    #TODO: change the 0 in all the marker pose indexing to an aruco id variable
    def step(self, action):
        
        # Get initial pose of the marker before moving to help calculate reward after moving
        start_marker_pose = self.find_marker_pose(marker_id=self.marker_id)
        
        try:
            state = self.gripper.move(action)
        except DynamixelServoError as error:
            # handle what to do if the gripper is unrecoverably gone wrong - i.e. save data and fail gracefully
            logging.error(error)
            exit()

        final_marker_pose = self.find_marker_pose(marker_id=self.marker_id)
        
        final_marker_yaw = -1
        if final_marker_pose is not None:
            final_marker_yaw = final_marker_pose[1][2]

        state.append(final_marker_yaw)
        
        reward, done = self.reward_function(self.target_angle, start_marker_pose, final_marker_pose)


        truncated = False #never truncate the episode but here for completion sake
        return state, reward, done, truncated