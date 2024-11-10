import logging
from enum import Enum
import random
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



class FourFingerRotation(FourFingerTask):

    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        self.port = gripper_config.touch_port
        self.num_sensors = gripper_config.num_touch_sensors
        super().__init__(env_config, gripper_config)
        if self.touch_config == True:
            # Initialise Touch Sensors
            print("Starting server...")
            self.tactile_server = server.Server(port=self.port, baudrate=921600)
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
        
    def get_values(self, host='localhost', server_port=65432):
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
        []
        object_orientation = (self._get_poses().get('object'))['orientation']

        # Constant Angle Goal Selection
        # new_goal = object_orientation[2] + random.choice([90,-90]) # +ve = CW, -ve = CCW
        # if new_goal > 360:
        #     new_goal = new_goal-360
        # elif new_goal < 0:
        #     new_goal = 360 - abs(new_goal)

        # Random Angle Goal Generation
        new_goal = random.randrange(0,361,1)
        print("new goal" , new_goal)
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
        # print('HERE')
        # print(cube_pose)
        #Output ={'position': array([  2.446419  ,   7.76192778, 285.87887697]), 'orientation': [179.44675100439056, 354.68999719345857, 178.36348896387162], 'r_vec': array([[-0.04422488, -3.04869736, -0.01673843]])}

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
            reward: Dependedent on the reward function.
                -Function 1: Distance-to-Goal reward function
                -Function 2: Delta Difference to goal reward function
                -Function 3: Combined Reward Function
            done: True if the goal is reached.
        """
        self.goal_reward = 300
        Precision_tolerance = 15
        touch_reward = 0
        touch_threshold = 1
        done = False
        logging.debug(previous_environment_info['poses']['object']['orientation'])
        
        previous_yaw = previous_environment_info['poses']['object']['orientation'][2]
        previous_yaw_diff = self.rotation_min_difference(self.goal[0], previous_yaw)
        current_yaw = current_environment_info['poses']['object']['orientation'][2]
        current_yaw_diff = self.rotation_min_difference(self.goal[0], current_yaw)

        #Touch-based reward
        if self.touch_config == True:
            print("Getting touch data in reward function")
            print("Max values after step: ", self.tactile_server.max_values)
            # Do reward based on touch sensor values
            for i in range(self.num_sensors):
                delta_touch = self.tactile_server.max_values[i] - self.sensor_baselines[i]
                if delta_touch < touch_threshold:
                    continue
                else:
                    touch_reward += 1
                    print("Touch Reward: ", touch_reward)
            # Reset the max values after each step
            self.tactile_server.max_values = self.sensor_baselines

        ##### Function 1
        # # Distance-to-Goal reward function
        # reward = round(-current_yaw_diff, 2)
        # # Reward set ot 0 if no cube no move
        # if abs(current_yaw_diff - previous_yaw_diff)<5:
        #     reward = 0
        #     print(reward)
        #     return reward, done
        #####

        ##### Function 2
        # # Delta Difference to goal reward function
        # delta = ((previous_yaw_diff - current_yaw_diff)/previous_yaw_diff) * 100
        # reward = round(delta, 2)
        # if reward < -100:
        #     reward = -100
        # # Negatively rewards for not moving the cube
        # if abs(delta) < 1:
        #     reward = -10
        #####
           
        ##### Function 3
        # Combined Reward Function
        A = 0.1 # Distance Coeffecient
        B = 1 # Delta Coefficient
        # Delta
        delta = ((previous_yaw_diff - current_yaw_diff)/previous_yaw_diff) * 100
        delta_reward = round(delta, 2)
        if delta_reward < -100:
            delta_reward = -100
        elif abs(delta_reward) < 1:
            delta_reward = 0
        # Distance
        distance_reward = round((-current_yaw_diff+180), 2)
        if abs(current_yaw_diff - previous_yaw_diff) < 1:
            distance_reward = 0
        reward = round((A*distance_reward) + (B*delta_reward), 2)
        #####

        if current_yaw_diff <= Precision_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = self.goal_reward
        print(f"Reward: ",reward)
        return reward, done
    
class FourFingerRotationSuspended(FourFingerRotation):
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
        self.elevator.move(self.elevator_max) # Raise Elevator
        self.gripper.move([2048,2200,2350,2048,2200,2350,2048,2200,2350,2048,2200,2350]) #Grasp Cube
        self.elevator.move(self.elevator_min)
        
        
    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False
        self.goal_reward = 300

        reward = 1
        return reward, done