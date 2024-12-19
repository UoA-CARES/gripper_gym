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

        self.goal_min = [40.0, 40.0]
        self.goal_max = [120.0, 120.0]

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
        
    def _pose_to_state(self, pose):
        state = []
        position = pose["position"]
        state.append(position[0] - self.reference_position[0])  # X
        state.append(position[1] - self.reference_position[1])  # Y
        return state

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
        state += self._pose_to_state(environment_info["poses"]["object"])

        # Goal
        state += environment_info["goal"]

        # Touch Sensor Values
        if self.touch_config == True:
            if self.tactile_server.error == True:
                print("Touch sensor server error. Rebooting server...")
                self.tactile_server.stop()
                time.sleep(5)
                self.tactile_server = server.Server(port=self.port, baudrate=921600, socket_port=self.socket_port)
                self.server_thread = threading.Thread(target=self.tactile_server.start)
                self.server_thread.daemon = True
                self.server_thread.start()
                time.sleep(5)
                while not self.tactile_server.server_ready:
                    time.sleep(1)
                self.tactile_server.error = False
            # for val in self.tactile_server.max_values:
            #     state += [val]
                
        logging.debug("State: ", state)
        return [round(val, 2) for val in state]

    def _render_environment(self, state, environment_state):
        # Get base rendering of the four-finger environment
        image = super()._render_environment(state, environment_state)

        image = cv2.rotate(self.camera.get_frame(), cv2.ROTATE_180) if self.is_inverted else self.camera.get_frame()

        image = cv2.undistort(
            image, self.camera.camera_matrix, self.camera.camera_distortion
        )

        # Draw the goal boundry for the translation task
        bounds_color = (255, 0, 0)
        bounds_min_x, bounds_min_y = utils.position_to_pixel(
            self.goal_min, self.reference_position, self.camera.camera_matrix
        )
        bounds_max_x, bounds_max_y = utils.position_to_pixel(
            self.goal_max, self.reference_position, self.camera.camera_matrix
        )
        cv2.rectangle(
            image,
            (int(bounds_min_x), int(bounds_min_y)),
            (int(bounds_max_x), int(bounds_max_y)),
            bounds_color,
            2,
        )
        
        # Draw reference marker
        marker_pos = self.reference_position
        marker_pixel = utils.position_to_pixel([0,0,0], marker_pos, self.camera.camera_matrix)
        cv2.circle(image, marker_pixel, 5, (255, 0, 0), -1)

        cv2.putText(
            image,
            f"Reference",
            marker_pixel,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,0,0),
            2,
            cv2.LINE_AA,
        )

        # Draw object positions
        object_color = (0, 255, 0)

        # Draw object's current position
        current_object_pose = environment_state["poses"]["object"]
        image, current_object_pixel = self._draw_circle(
            image,
            current_object_pose["position"][0:2],
            [0, 0, current_object_pose["position"][2]],
            object_color,
        )

        cv2.putText(
            image,
            "Current",
            (current_object_pixel[0]+self.noise_tolerance, current_object_pixel[1]+self.noise_tolerance), # Text location adjusted for circle size
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            object_color,
            2,
        )

        # Draw object's previous position
        previous_object_pose = self.previous_environment_info["poses"]["object"]
        image, previous_object_pixel = self._draw_circle(
            image,
            previous_object_pose["position"][0:2],
            [0, 0, previous_object_pose["position"][2]],
            object_color,
        )

        cv2.putText(
            image,
            "Previous",
            (previous_object_pixel[0]+self.noise_tolerance, previous_object_pixel[1]+self.noise_tolerance),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            object_color,
            2,
        )

        # Draw line from previous to current
        cv2.line(image, current_object_pixel, previous_object_pixel, (255, 0, 0), 2)

        # Draw goal position - note the reference Z is relative to the Marker ID of the target for proper math purposes
        goal_color = (0, 0, 255)
        goal_reference_position = [
            self.reference_position[0],
            self.reference_position[1],
            current_object_pose["position"][2],
        ]
        image, goal_pixel = self._draw_circle(
            image,
            self.goal,
            self.reference_position,#goal_reference_position,
            goal_color,
        )

        # Draw line from object to goal
        cv2.line(image, current_object_pixel, goal_pixel, (255, 0, 0), 2)

        # Draw goal position
        goal_color = (0, 0, 255)
        goal_reference_position = [
            self.reference_position[0],
            self.reference_position[1],
            environment_state["poses"]["object"]["position"][2],
        ]
        image, goal_pixel = self._draw_circle(
            image,
            self.goal,
            self.reference_position,#goal_reference_position,
            goal_color,
        )


        return image
    
    def _draw_circle(self, image, position, reference_position, color):
        pixel_location = utils.position_to_pixel(
            position,
            reference_position,
            self.camera.camera_matrix,
        )
        # Circle size now reflects the "Close enough" to goal tolerance
        cv2.circle(image, pixel_location, self.noise_tolerance, color, -1)
        return image, pixel_location
    
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
        Calculate the center point of a cube base on the detected markers.
        Args:
            marker_poses (dict): A dictionary containing the poses of the detected ArUco markers.
        Returns:
            dict: A dictionary containing the position and orientation of the cube.
        """
        cube_ids = [1,2,3,4,5,6]
        detected_ids = [ids for ids in marker_poses]

        cube_marker_ids = [id for id in cube_ids if id in detected_ids]

        if len(cube_marker_ids) == 0:
            # If no cube marker detected, return a default pose assuming the cube has been dropped
            return  {'position': np.array([1.0, 150.0, 200.0]), 'orientation': [1.0, 1.0, 1.0]}
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
            self.tactile_server.baseline_values = self.get_values(self.socket_port)
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
        #TODO
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
                    delta_touch = self.tactile_server.max_values[i] - self.tactile_server.baseline_values[i]
                    if delta_touch < touch_threshold:
                        continue
                    else:
                        num_touch += 1
                reward += num_touch*touch_reward
                reward = round(reward, 2)
                print("Number of touch sensors triggered: ", num_touch, "Reward: ", num_touch*touch_reward)
                # Reset the max values after each step
                self.tactile_server.max_values = self.tactile_server.baseline_values

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
        self.noise_tolerance = env_config.noise_tolerance
        self.aruco_detector = STagDetector(marker_size=env_config.marker_size, library_hd=11)
        super().__init__(env_config, gripper_config)
        self.elevator_device_name = env_config.elevator_device_name
        self.elevator_baudrate = env_config.elevator_baudrate
        self.elevator_servo_id = env_config.elevator_servo_id
        self.elevator_min = env_config.elevator_limits[0] # Lowered Elevator Position
        self.elevator_max = env_config.elevator_limits[1] # Extended Elevator Position

        self.total_moves = 0

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
        self.iscubedropped = False
        self.init_elevator()
        self.elevator.enable_torque()

        # TODO implement object centred check
        self.elevator.move(self.elevator_min) # Lower Elevator
        self.gripper.wiggle_home() # Home Gripper 
        # Opening Grasp
        if self.gripper_config.touch:
            time.sleep(1)
            self.tactile_server.baseline_values = self.get_values(self.socket_port)
            print("Baseline values updated", self.tactile_server.baseline_values)
        self.elevator.move(self.elevator_max) # Raise Elevator
        self.gripper.move([2048,2200,2350,2048,2200,2350,2048,2200,2350,2048,2200,2350]) #Grasp Cube
        self.elevator.move(self.elevator_min)
        
        
    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False
        reward = 0
        self.goal_reward = 400
        height_threshold = 200
        if self.iscubedropped:
            current_height = 230
        else:
            current_height = current_environment_info['poses']['object']['position'][2]
        print("Current Height: ", current_height)

        # Get current and previous distance to goal
        target_pose = current_environment_info["goal"]
        target_pose = [target_pose[0]+self.reference_position[0], target_pose[1]+self.reference_position[1]]
        current_object_pose = current_environment_info["poses"]["object"]["position"][0:2]
        previous_object_pose = previous_environment_info["poses"]["object"]["position"][0:2]

        # Calculate the distance to the goal
        previous_goal_distance = math.dist(target_pose, previous_object_pose)
        current_goal_distance = math.dist(target_pose, current_object_pose)
        print("Previous Goal Distance: ", previous_goal_distance, "Current Goal Distance: ", current_goal_distance)

        # Touch-based reward oustside of height threshold check
        if self.touch_config == True:
                num_touch = 0
                touch_threshold = 1
                touch_reward = 50
                print("Getting touch data in reward function")
                print("Max values after step: ", self.tactile_server.max_values)
                # Do reward based on touch sensor values
                for i in range(self.num_sensors):
                    delta_touch = self.tactile_server.max_values[i] - self.tactile_server.baseline_values[i]
                    if delta_touch < 0:
                        continue
                    if delta_touch < touch_threshold:
                        continue
                    else:
                        num_touch += 1
                # Reward based on number of touch sensors triggered
                reward += num_touch*touch_reward
                print("Number of touch sensors triggered: ", num_touch, "Reward: ", num_touch*touch_reward)
                # Reset the max values after each step
                self.tactile_server.max_values = self.tactile_server.baseline_values

        ######## Combined Distance-Delta Reward Function
        # A= 0.1 # Distance Coeffecient
        # B = 1  # Delta Coefficient
        # print("Current Height: ", current_height)

        # # Calculate the delta change in distance to the goal
        # delta = previous_goal_distance - current_goal_distance 
        # delta = (delta/previous_goal_distance)
        # if abs(delta) < 0.1:
        #     delta = 0 
        # delta_reward = (delta*self.goal_reward) if delta >= -1 else -1*self.goal_reward
        # print("Delta Reward: ", delta_reward, "Delta: ", delta)

        # # Calculate the distance reward
        # distance_reward = round((-current_goal_distance+100),2)
        # print("Distance Reward: ", A*distance_reward)

        # if current_height < height_threshold:
        #     reward += round((A*distance_reward) + (B*delta_reward), 2)
        
        ##### Staged Reward Function
        # Check if cube above height threshold
        if current_height < height_threshold:
            # Stage 1
            reward += 100

            delta = previous_goal_distance - current_goal_distance
            print("Delta: ", delta)

            if self.total_moves < 50:
                #Stage 2
                if -10 < delta < 10:
                    # Did not move cube
                    reward -= 50
                else:
                    # Did move cube
                    reward += 50
                    self.total_moves += 1
                    print("Moved", self.total_moves)
            else:
                #Stage 3
                raw_reward = (delta/previous_goal_distance)
                print("Raw Reward: ", raw_reward*self.goal_reward)
                reward += round((raw_reward*self.goal_reward), 2)
        else:
            # Cube fallen threshold
            reward = -100


        # Check if the goal is reached
        if current_goal_distance < self.noise_tolerance and current_height < height_threshold:
                reward = self.goal_reward
                logging.info(f"Goal Reached!")
        reward = round(reward, 2)
        print(f"Total Reward: ",reward)
        return reward, done
    