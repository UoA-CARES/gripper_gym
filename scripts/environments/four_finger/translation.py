import logging
from enum import Enum
from random import randrange
import time

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
from cares_lib.dynamixel.Servo import Servo
from cares_lib.touch_sensors import server
import threading
import socket
import ast
import dynamixel_sdk as dxl



class FourFingerTranslation(FourFingerTask):

    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        self.port = gripper_config.touch_port
        self.num_sensors = gripper_config.num_touch_sensors
        self.socket_port = gripper_config.socket_port

        self.boundary_size = 75
        self.image_size = [640,480]
        self.image_centre = [(self.image_size[0]/2), (self.image_size[1]/2)]
        self.goal_min = [(self.image_centre[0]-self.boundary_size),(self.image_centre[1]-self.boundary_size)]
        self.goal_max = [(self.image_centre[0]+self.boundary_size),(self.image_centre[1]+self.boundary_size)]

        super().__init__(env_config, gripper_config)
        if self.touch_config == True:
            # Initialise Touch Sensors
            print("Starting server...")
            self.tactile_server = server.Server(port=self.port, baudrate=921600, socket_port=self.socket_port)
            self.server_thread = threading.Thread(target=self.tactile_server.start)
            self.server_thread.daemon = True
            self.server_thread.start()
            print("Server started in separate thread.")
            while not self.tactile_server.server_ready:
                time.sleep(0.5)
            print("Server ready.")

            # Create Sensor data object
            self.sensor_baselines = self.tactile_server.baseline_values
            print("Baselines: ", self.sensor_baselines)
        
    def get_values(self, server_port, host='localhost'):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect((host, server_port))
                data = client_socket.recv(1024).decode('utf-8')
                data = ast.literal_eval(data)
                return data
        except ConnectionRefusedError:
            return "Failed to connect to the server."
        except ConnectionResetError:
            return "Connection to the server was reset."

    # overriding method
    def _choose_goal(self):
        """
        Chooses a goal for the current environment state. Currently always rotates to 90 clockwise
        Returns:
            Chosen goal.
        """
        x1, y1 = self.goal_min
        x2, y2 = self.goal_max

        goal_x = randrange(x1, x2)
        goal_y = randrange(y1, y2)
        
        print("New Goal: ", goal_x, goal_y)
        return [goal_x, goal_y]
        

    def _environment_info_to_state(self, environment_info):
        state = []

        # Servo Angles - Steps
        state += environment_info["gripper"]["positions"]

        # Object position - XY 
        for i in range(0,2):
            state += [round(environment_info["poses"]["object"]["position"][i],2)]

        # Goal
        state += environment_info["goal"]

        # Touch Sensor Values
        if self.touch_config == True:
            for val in self.tactile_server.max_values:
                state += [val]
                
        logging.debug("State: ", state)
        return [round(val, 2) for val in state]

    def _render_environment(self, state, environment_state):
        # Get base rendering of the four-finger environment
        image = super()._render_environment(state, environment_state)

        image = cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame()

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        # Draw the goal boundary, X640 Y480
        boundary_colour = (255,0,0)
        print("Centre: ", self.image_centre, "Top Left: ", self.goal_min, "Bottom Right: ", self.goal_max)
        # Draw the goal
        goal_colour = (0,0,255)
        cv2.circle(image, self.goal, 10, goal_colour, -1)
        # Draw the boundary box
        cv2.rectangle(
            image,
            (int(self.goal_min[0]), int(self.goal_min[1])),             # Top left corner
            (int(self.goal_max[0]), int(self.goal_max[1])),         # Bottom right corner
            boundary_colour,
            2,
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
        

class FourFingerTranslationFlat(FourFingerTranslation):
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
        if self.gripper_config.touch:
            time.sleep(1)
            self.tactile_server.baseline_values = self.get_values(self.gripper_config.socket_port)
            print("Baseline values updated", self.tactile_server.baseline_values)


    # overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        """
        Computes the reward based on the target goal and the change in yaw.

        Returns:
            reward: Dependedent on the reward function.
                -Function 1: Distance-to-Goal reward function
                -Function 2: Delta Difference to goal reward function
                -Function 3: Combined Reward Function
            done: True if the goal is reached.
        """
        done = False
        touch_threshold = 1
        touch_reward = 50
        num_touch = 0
        # Combined Distance-Delta Reward Function
        # Combined Reward Function
        A = 0.1 # Distance Coeffecient
        B = 1 # Delta Coefficient
        # Delta
        delta_reward = 0
        # Distance
        distance_reward = 0
        #####

        #Touch-based reward
        if self.touch_config == True:
            if delta_reward != 0 and distance_reward != 0:
                print("Getting touch data in reward function")
                print("Max values after step: ", self.tactile_server.max_values)
                # Do reward based on touch sensor values
                for i in range(self.num_sensors):
                    delta_touch = self.tactile_server.max_values[i] - self.sensor_baselines[i]
                    if delta_touch < touch_threshold:
                        continue
                    else:
                        num_touch += 1
                reward += num_touch*touch_reward
                reward = round(reward, 2)
                print("Number of touch sensors triggered: ", num_touch, "Reward: ", num_touch*touch_reward)
                # Reset the max values after each step
                self.tactile_server.max_values = self.sensor_baselines

        reward = 1
        print(f"Total Reward: ",reward)
        return reward, done
    
class FourFingerTranslationSuspended(FourFingerTranslation):
    def __init__(
        self, 
        env_config: GripperEnvironmentConfig, 
        gripper_config: GripperConfig
    ):
        self.env_config = env_config
        self.gripper_config = gripper_config
        self.aruco_detector = STagDetector(marker_size=env_config.marker_size, library_hd=11)
        super().__init__(env_config, gripper_config)
        self.elevator_device_name = env_config.elevator_device_name
        self.elevator_baudrate = env_config.elevator_baudrate
        self.elevator_servo_id = env_config.elevator_servo_id
        self.elevator_min = env_config.elevator_limits[0] # Lowered Elevator Position
        self.elevator_max = env_config.elevator_limits[1] # Extended Elevator Position

    def init_elevator(self):
        self.elevator_port_handler = dxl.PortHandler(self.elevator_device_name)
        self.elevator_packet_handler = dxl.PacketHandler(2)
        self.elevator = Servo(
            self.elevator_port_handler, 
            self.elevator_packet_handler, 
            2, 
            self.elevator_servo_id, 
            1, 
            200, 
            200, 
            self.elevator_min, 
            self.elevator_max, 
            model="XL330-M077-T"
            )

        if not self.elevator_port_handler.openPort():
            error_message = f"Failed to open port {self.elevator_device_name}"
            logging.error(error_message)    
            raise IOError(error_message)
        logging.debug(f"Succeeded to open port {self.elevator_device_name}")

        if not self.elevator_port_handler.setBaudRate(self.elevator_baudrate):
            error_message = f"Failed to change the baudrate to {self.elevator_baudrate}"
            logging.error(error_message)
            raise IOError(error_message)
        logging.debug(f"Succeeded to change the baudrate to {self.elevator_baudrate}")
    
    def _reset(self):
        self.init_elevator()
        self.elevator.enable_torque()

        # TODO implement object centred check
        self.elevator.move(self.elevator_min) # Lower Elevator
        self.gripper.wiggle_home() # Home Gripper 
        # Opening Grasp
        if self.gripper_config.touch:
            time.sleep(1)
            self.tactile_server.baseline_values = self.get_values(self.gripper_config.socket_port)
            print("Baseline values updated", self.tactile_server.baseline_values)
        self.elevator.move(self.elevator_max) # Raise Elevator
        self.gripper.move([2048,2200,2350,2048,2200,2350,2048,2200,2350,2048,2200,2350]) #Grasp Cube
        self.elevator.move(self.elevator_min)
        
        
    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False
        reward = 0
        self.goal_reward = 400
        height_threshold = 350
        current_height = current_environment_info['poses']['object']['position'][2]
        print("Current Height: ", current_height)
        if current_height < height_threshold:
            reward += self.goal_reward
        if self.touch_config == True:
            num_touch = 0
            touch_threshold = 2
            touch_reward = 50
            print("Getting touch data in reward function")
            print("Max values after step: ", self.tactile_server.max_values)
            # Do reward based on touch sensor values
            for i in range(self.num_sensors):
                delta_touch = self.tactile_server.max_values[i] - self.sensor_baselines[i]
                if delta_touch < 0:
                    continue
                if delta_touch < touch_threshold:
                    continue
                else:
                    num_touch += 1
            # Reward based on number of touch sensors triggered
            if current_height < height_threshold:
                reward += num_touch*touch_reward
                reward = round(reward, 2)
            print("Number of touch sensors triggered: ", num_touch, "Reward: ", num_touch*touch_reward)
            # Reset the max values after each step
            self.tactile_server.max_values = self.sensor_baselines

        print(f"Total Reward: ",reward)
        return reward, done