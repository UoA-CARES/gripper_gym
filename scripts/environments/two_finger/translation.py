import logging
import math
from random import randrange
import numpy as np
import cv2

from cares_lib.dynamixel.gripper_configuration import GripperConfig
from configurations import GripperEnvironmentConfig
from environments.two_finger.two_finger import TwoFingerTask

from cares_lib.dynamixel.Servo import Servo, DynamixelServoError
from cares_lib.dynamixel.Gripper import GripperError
import tools.utils as utils

class TwoFingerTranslation(TwoFingerTask):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        self.noise_tolerance = env_config.noise_tolerance

        # These bounds are respective to the reference marker in Environment
        self.goal_min = [-30.0, 60.0]
        self.goal_max = [120.0, 110.0]

        logging.debug(
            f"Goal Min: {self.goal_min} Goal Max: {self.goal_max} Tolerance: {self.noise_tolerance}"
        )

        super().__init__(env_config, gripper_config)

    # overriding method
    def _choose_goal(self):
        x1, y1 = self.goal_min
        x2, y2 = self.goal_max

        goal_x = randrange(x1, x2)
        goal_y = randrange(y1, y2)

        return [goal_x, goal_y]

    # overriding method
    def _environment_info_to_state(self, environment_info):
        state = []

        # Servo Angles - Steps
        # state += environment_info["gripper"]["positions"]

        # Servo Velocities - Steps per second
        if self.action_type == "velocity":
            state += environment_info["gripper"]["velocities"]

        # Servo + Two Finger Tips - X Y mm
        for i in range(1, self.gripper.num_motors + 3):
            servo_position = environment_info["poses"]["gripper"][i]
            state += self._pose_to_state(servo_position)

        # Object - X Y mm
        state += self._pose_to_state(environment_info["poses"]["object"])

        # Goal State - X Y mm
        state += self.goal

        return state

    # overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        object_previous = previous_environment_info["poses"]["object"]["position"][0:2]
        object_current = current_environment_info["poses"]["object"]["position"][0:2]

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        goal_progress = goal_distance_before - goal_distance_after

        # The following step might improve the performance.

        # goal_before_array = goal_before[0:2]
        # delta_changes   = np.linalg.norm(target_goal - goal_before_array) - np.linalg.norm(target_goal - goal_after_array)
        # if -self.noise_tolerance <= delta_changes <= self.noise_tolerance:
        #     reward = -10
        # else:
        #     reward = -goal_difference
        #     #reward = delta_changes / (np.abs(yaw_before - target_goal))
        #     #reward = reward if reward > 0 else 0

        # For Translation. noise_tolerance is 15, it would affect the performance to some extent.
        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            done = True
            reward = 500
        else:
            reward += goal_progress

        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        return reward, done


class TwoFingerTranslationFlat(TwoFingerTranslation):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)

    # overriding method
    def _reset(self):
        self.gripper.wiggle_home()

    def _get_poses(self):
        """
        Gets the current state of the environment using the Aruco markers.

        Returns:
        dict : A dictionary containing the poses of the gripper and object markers.

        gripper: X-Y-Z-RPY Servos + X-Y-Z-RPY Finger-tips
        object: X-Y-Z-RPY Object
        """
        poses = {}

        # Servos + Finger Tips (2)
        num_gripper_markers = self.gripper.num_motors + 2

        # Gripper markers + Object (1)
        num_markers = num_gripper_markers + 1

        # maker_ids match servo ids (counting from 1)
        marker_ids = [id for id in range(1, num_markers + 1)]

        marker_poses = self._get_marker_poses(marker_ids)
        
        poses["gripper"] = dict(
            [i, marker_poses[i]] for i in range(1, num_gripper_markers + 1)
        )

        object_marker_id = num_markers
        poses["object"] = marker_poses[object_marker_id]

        return poses

    


class TwoFingerTranslationSuspended(TwoFingerTranslation):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)
        led = id = 5
        self.max_value = 3500
        self.min_value = 0
        servo_type = "XL330-M077-T"
        speed_limit = torque_limit = 150
        
        try:
            self.lift_servo = Servo(self.gripper.port_handler, self.gripper.packet_handler, self.gripper.protocol, id, led,
                                            torque_limit, speed_limit, self.max_value,
                                            self.min_value, servo_type)
            self.lift_servo.enable()
        except (GripperError, DynamixelServoError) as error:
            raise GripperError(f"Gripper#{self.gripper_id}: Failed to initialise lift servo") from error
        
    
    # overriding method
    def _reset(self):
        self.gripper.home()
        self._grab_cube()
    
    def _lift_up(self):
        self.lift_servo.move(self.max_value,timeout=3)

    def _lift_down(self):
        self.lift_servo.move(self.min_value)

    def _grab_cube(self):
        self._lift_up()
        # check cube is above line?
        # reboot lift servo if needed
        self.gripper.move([512,362,512,662])
        self._lift_down()


    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False
        line = 90

        reward = 0

        # y = self._pose_to_state(current_environment_info["poses"]["object"])[1]
        y = current_environment_info["poses"]["object"]["position"][1]

        reward += 1 if y < line else -1
        print(y, reward)

        return reward, done
    
    def _get_poses(self):
        """
        Gets the current state of the environment using the Aruco markers.

        Returns:
        dict : A dictionary containing the poses of the gripper and object markers.

        gripper: X-Y-Z-RPY Servos + X-Y-Z-RPY Finger-tips
        object: X-Y-Z-RPY Object
        """
        poses = {}

        # Servos + Finger Tips (2)
        num_gripper_markers = self.gripper.num_motors + 2

        # maker_ids match servo ids (counting from 1)
        marker_ids = [id for id in range(1, num_gripper_markers + 1)]

        marker_poses = self._get_marker_poses(marker_ids)

        poses["gripper"] = dict(
            [i, marker_poses[i]] for i in range(1, num_gripper_markers + 1)
        )

        poses["object"] = self._get_cube_pose(marker_poses)

        return poses
    
    def _get_cube_pose(self, marker_poses):
        """
        Calculate the center point of a cube base on the detected markers.
        Args:
            marker_poses (dict): A dictionary containing the poses of the detected ArUco markers.
        Returns:
            dict: A dictionary containing the position and orientation of the cube.
        """
        cube_ids = [7,8,9,10,11,12]
        detected_ids = [ids for ids in marker_poses]

        cube_marker_ids = [id for id in cube_ids if id in detected_ids]

        if len(cube_marker_ids) == 0:
            # If no cube marker detected, return a default pose assuming the cube has been dropped
            return  {'position': np.array([1.0, 150.0, 1.0]), 'orientation': [1.0, 1.0, 1.0]}
        else:
            # Calculate the cube centers for the marker IDs present in both cube_ids and detected_ids
            cube_centers = [self._calculate_cube_center(marker_poses[id]["position"], marker_poses[id]["r_vec"])
            for id in cube_marker_ids]

            # Calculate the final cube center by averaging
            cube_centers = np.array(cube_centers)
            cube_center = np.mean(cube_centers, axis=0)

        return {'position': cube_center, 'orientation': [1.0, 1.0, 1.0]}
    
    def _calculate_cube_center(self, marker_position, r_vec, cube_size=50):
        """
        Calculate the center point of a cube given the position and orientation of one face.  
        Args:
            marker_position (numpy.ndarray): A 1D array of length 3 representing the x, y, z coordinates of the center of the face.
            r_vec (numpy.ndarray): A 1D array of length 3 representing the row, pitch, yaw angles (in radians) of the face.
            cube_size (int): The size of the cube (default is 50).
        Returns:
            numpy.ndarray: A 1D array of length 3 representing the x, y, z coordinates of the center of the cube.
        """

        # Calculate the rotation matrix from the Rodrigues vector
        rotation_matrix, _ = cv2.Rodrigues(r_vec)

        # Calculate the offset from the face center to the cube center
        offset = np.dot(rotation_matrix, np.array([0, 0, cube_size / 2]))

        # Calculate the cube center
        cube_center = marker_position - offset

        return cube_center
    
    # overriding method
    def _environment_info_to_state(self, environment_info):
        state = []

        # Servo Velocities - Steps per second
        if self.action_type == "velocity":
            state += environment_info["gripper"]["velocities"]

        # Servo + Two Finger Tips - X Y mm
        for i in range(1, self.gripper.num_motors + 3):
            servo_position = environment_info["poses"]["gripper"][i]
            state += self._pose_to_state(servo_position)

        # Object - X Y mm
        state += self._pose_to_state(environment_info["poses"]["object"])

        # Goal State - Y mm
        state += [90]

        return state
    

    def _render_envrionment(self, state, environment_state):

        image = self.camera.get_frame()

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        bounds_color = (0, 255, 0)

        goal_line_pixel = utils.position_to_pixel(
            [0, state[-1], 0],
            self.reference_position,
            self.camera.camera_matrix,
        )

        cv2.line(image, (0, goal_line_pixel[1]), (640, goal_line_pixel[1]), bounds_color, 2)

        # Draw object position
        object_color = (0, 255, 0)
        object_pose = environment_state["poses"]["object"]
        object_pixel = utils.position_to_pixel(
            object_pose["position"],
            [0, 0, object_pose["position"][2]],
            self.camera.camera_matrix,
        )
        cv2.circle(image, object_pixel, 9, object_color, -1)

        num_gripper_markers = self.gripper.num_motors + 2

        base_index =  4 if self.action_type == "velocity" else 0

        for i in range(0, num_gripper_markers):
            x = state[base_index + i * 2]
            y = state[base_index + i * 2 + 1]

            position = [
                x,
                y,
                environment_state["poses"]["gripper"][i + 1]["position"][2],
            ]

            reference = self.reference_position
            reference[2] = position[2]
            marker_pixel = utils.position_to_pixel(
                position,
                reference,
                self.camera.camera_matrix,
            )
            cv2.circle(image, marker_pixel, 9, (0, 255, 0), -1)

            cv2.putText(
                image,
                f"{i+1}",
                marker_pixel,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("State Image", image)
        cv2.waitKey(10)
