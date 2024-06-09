import logging
from enum import Enum
import random

import numpy as np
import cv2
import math
import tools.utils as utils
from configurations import GripperEnvironmentConfig
from environments.four_finger.four_finger import FourFingerTask
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.vision.STagDetector import STagDetector
from cares_lib.dynamixel.Gripper import GripperError
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_lib.touch_sensors.sensor import Sensor



class FourFingerRotation(FourFingerTask):

    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        
        super().__init__(env_config, gripper_config)

    # overriding method
    def _choose_goal(self):
        """
        Chooses a goal for the current environment state. Currently always rotates to 90 clockwise
        Returns:
            Chosen goal.
        """
        object_orientation = (self._get_poses().get('object'))['orientation']
        new_goal = object_orientation[2] + 90 # + = CW
        if new_goal > 360:
            new_goal = new_goal-360
        return [new_goal]
        

    def _environment_info_to_state(self, environment_info):
        state = []

        # Servo Angles - Steps
        state += environment_info["gripper"]["positions"]

        # Object position - XY 
        for i in range(0,2):
            state += [environment_info["poses"]["object"]["position"][i]]

        # Object orientation
        state += [environment_info["poses"]["object"]["orientation"][2]]

        # Goal
        state += environment_info["goal"]
        
        return [round(val, 2) for val in state]

    def _render_environment(self, state, environment_state):
        # Get base rendering of the four-finger environment
        image = super()._render_environment(state, environment_state)

        image = cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame()

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        #TODO 
        # Image Size X640 Y480
        position = environment_state['poses']['object']['position']
        pixel_x = self.camera.camera_matrix[0,0] * position[0]/320 + self.camera.camera_matrix[0,2]
        pixel_y = self.camera.camera_matrix[1,1] * position[1]/240 + self.camera.camera_matrix[1,2]
        centre = [round(pixel_x), round(pixel_y)]

        # TODO put arrow_end calculation into function
        yaw = environment_state['poses']['object']['orientation'][2]
        lineSize = 35
        arrow_end_x = position[0] + (math.sin(math.radians(yaw)) * lineSize)
        arrow_end_x = self.camera.camera_matrix[0,0] * arrow_end_x/320 + self.camera.camera_matrix[0,2]
        arrow_end_y = position[1] - (math.cos(math.radians(yaw)) * lineSize)
        arrow_end_y = self.camera.camera_matrix[1,1] * arrow_end_y/240 + self.camera.camera_matrix[1,2]
        arrow_end_axis = [round(arrow_end_x), round(arrow_end_y)]

        arrow_end_x = position[0] + (math.sin(math.radians(self.goal[0])) * lineSize)
        arrow_end_x = self.camera.camera_matrix[0,0] * arrow_end_x/320 + self.camera.camera_matrix[0,2]
        arrow_end_y = position[1] - (math.cos(math.radians(self.goal[0])) * lineSize)
        arrow_end_y = self.camera.camera_matrix[1,1] * arrow_end_y/240 + self.camera.camera_matrix[1,2]
        arrow_end_goal = [round(arrow_end_x), round(arrow_end_y)]
        
        # Places a circle at the centre of the cube marker
        cv2.circle(image, centre, 5, (0,0,255), -1)
        # Draws an arrow of the markers X axis reference, this is the axis which the angle refers to. The -Y axis is seen as 0/360 degrees.
        cv2.arrowedLine(image, centre, arrow_end_axis, (255,0,0), 3)
        # Draws an arrow of the markers desired X axis placement, i.e. the goal angle
        cv2.arrowedLine(image, centre, arrow_end_goal, (255,0,0), 3)

        cv2.putText(
                image,
                f"{'Current'}",
                arrow_end_axis,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        
        cv2.putText(
                image,
                f"{'Goal'}",
                arrow_end_goal,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return image
    
    def _get_poses(self):
        """
        Gets the current state of the environment using the Aruco markers.

        Returns:
        dict : A dictionary containing the pose of the object marker.
        object: X-Y-Z-RPY Object
        """
        poses = {}
        marker_poses = self._get_marker_poses(self.env_config.cube_ids)
        poses["object"] = self._get_cube_pose(marker_poses) # Converts marker pose into cube pose
        
        return poses
    
    def _get_cube_pose(self, marker_poses):
        """
        Returns the pose of the cube based on the detected marker
        Args:
            marker_poses (dict): A dictionary containing the poses of the detected Aruco markers
        Returns:
            array: An array containing the orientation of the cube.
        """

        cube_ids = [1,2,3,4,5,6]
        detected_ids = [id for id in cube_ids if id in marker_poses]
        cube_pose = (list(marker_poses.values()))[0]
        return cube_pose
        

class FourFingerRotationFlat(FourFingerRotation):
    def __init__(
        self, 
        env_config: GripperEnvironmentConfig, 
        gripper_config: GripperConfig
    ):
        self.env_config = env_config
        self.gripper_config = gripper_config
        self.aruco_detector = STagDetector(marker_size=env_config.marker_size, library_hd=11)
        super().__init__(env_config, gripper_config)
        
    def _reset(self):
        self.gripper.wiggle_home()


    # overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        """
        Computes the reward based on the target goal and the change in yaw.

        Returns:
            reward: reward = 10 if at goal, negative when rotated away from goal(max of -1), otherwise a fraction of the progress made to the goal.
                    reward = -1 if the cube is not rotated at all.
        """

        Precision_tolerance = 15
        done = False
        logging.debug(previous_environment_info['poses']['object']['orientation'])
        
        previous_yaw = previous_environment_info['poses']['object']['orientation'][2]
        previous_yaw_diff = self.rotation_min_difference(self.goal[0], previous_yaw)
        current_yaw = current_environment_info['poses']['object']['orientation'][2]
        current_yaw_diff = self.rotation_min_difference(self.goal[0], current_yaw)

        # Distance-to-Goal reward function
        # reward = round(-current_yaw_diff+90, 2)
        # # Reward set ot 0 if no cube no move
        # if abs(current_yaw_diff - previous_yaw_diff)<5:
        #     reward = 0
        #     print(reward)
        #     return reward, done

        # Delta Difference to goal reward function
        delta = previous_yaw_diff - current_yaw_diff


        

        if current_yaw_diff <= Precision_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 150
        print(reward)
        return reward, done
    
    def rotation_min_difference(self, a, b):
        """
        Formula that calculates the minimum difference between two angles.

        Args:
        a: First angle.
        b: Second angle.

        Returns:
            float: The minimum angular difference.
        """
        return min(abs(a - b), (360 + min(a, b) - max(a, b)))