import logging
import math
from random import randrange
import numpy as np
import cv2

import cv2
import tools.utils as utils
from configurations import GripperEnvironmentConfig
from environments.two_finger.two_finger import TwoFingerTask

from cares_lib.dynamixel.Servo import Servo, DynamixelServoError
from cares_lib.dynamixel.Gripper import GripperError
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.vision.STagDetector import STagDetector
from cares_lib.touch_sensors.sensor import Sensor
import tools.utils as utils
import dynamixel_sdk as dxl
import time


class TwoFingerTranslation(TwoFingerTask):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        self.noise_tolerance = env_config.noise_tolerance
        logging.debug(
            f"Goal Min: {self.goal_min} Goal Max: {self.goal_max} Tolerance: {self.noise_tolerance}"
        )

        super().__init__(env_config, gripper_config)
        self.env_config = env_config
        if env_config.touch == True:
            self.Touch = Sensor("/dev/ttyACM0", 921600)
            self.Touch.initialise()
            
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

        #Touch Sensor Values
        if self.env_config.touch:
            readings = self.Touch.get_pressure_readings()
            readings[:] = [x/10 for x in readings]
            state += readings
            print("State Input",readings)

        return state
    

    def _draw_circle(self, image, position, reference_position, color):
        pixel_location = utils.position_to_pixel(
            position,
            reference_position,
            self.camera.camera_matrix,
        )
        # Circle size now reflects the "Close enough" to goal tolerance
        cv2.circle(image, pixel_location, self.noise_tolerance, color, -1)
        return image, pixel_location
    

    def _render_environment(self, state, environment_state):
        # Get base rendering of the two-finger environment
        image = super()._render_environment(state, environment_state)

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
            goal_reference_position,
            goal_color,
        )

        # Draw line from object to goal
        cv2.line(image, current_object_pixel, goal_pixel, (255, 0, 0), 2)

        reward, _ = self._reward_function(
            self.previous_environment_info, self.current_environment_info
        )
        cv2.putText(
            image,
            f"Reward: {reward}",
            (goal_pixel[0]+self.noise_tolerance, goal_pixel[1]+self.noise_tolerance),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        # Draw circle highlighting goal_range
        pixel_location_goal = utils.position_to_pixel(
            self.goal,
            goal_reference_position,
            self.camera.camera_matrix,
        )
        cv2.circle(image, pixel_location_goal, 2*self.goal_range, (0, 255, 0), 2)

        return image


class TwoFingerTranslationFlat(TwoFingerTranslation):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        self.aruco_detector = STagDetector(marker_size=env_config.marker_size)
        # These bounds are respective to the reference marker in Environment
        self.goal_min = [-40.0, 70.0]
        self.goal_max = [100.0, 110.0]

        super().__init__(env_config, gripper_config)
        self.elevator_device_name = env_config.elevator_device_name
        self.elevator_baudrate = env_config.elevator_baudrate
        self.elevator_servo_id = env_config.elevator_servo_id
        self.goal_range = 70
        self.elevator_max = env_config.elevator_limits[0]
        self.elevator_min = env_config.elevator_limits[1]
        #TODO add instantiation of the elevator elevator servo here 

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
            self.elevator_max, 
            self.elevator_min, 
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

    # overriding method
    def _reset(self):
        self.init_elevator()
        self.elevator.enable_torque()
        self.elevator.set_operating_mode(4)

        # TODO implement object centred check
        self.gripper.move([312, 712, 512, 512])
        while not (self.gripper.is_home()):
            if self.elevator.current_goal_position() < (self.elevator_max-100):
                self.elevator.move(self.elevator_max)
            self.elevator.move(self.elevator_min)
            self.gripper.home()

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
    
    #overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        object_previous = self._pose_to_state(previous_environment_info["poses"]["object"])
        object_current = self._pose_to_state(current_environment_info["poses"]["object"])
        logging.debug(f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}")

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)
        
        logging.debug(f"Distance to Goal: {goal_distance_after}")

        # For Translation. noise_tolerance is 15, it would affect the performance to some extent.
        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 80
        elif goal_distance_after > self.goal_range:
            reward = 0
        else:
            reward = round((-goal_distance_after+self.goal_range),2)
        
        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )
        if self.env_config.touch:
            self.Touch.reset_pressure_readings()
        return reward, done

    


class TwoFingerTranslationSuspended(TwoFingerTranslation):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        self.aruco_detector = ArucoDetector(marker_size=env_config.marker_size)
        # These bounds are respective to the reference marker in Environment
        self.goal_min = [-20.0, 70]
        self.goal_max = [100.0, 105]

        super().__init__(env_config, gripper_config)
        self.max_value = 3500
        self.min_value = 0
        self.goal_line = 45
        self.bottom_line = 90 + abs(self.reference_position[1])

        self.goal_range = 50

        self._init_lift_servo(self.gripper)
        
    def _init_lift_servo(self, gripper):
        led = id = 5
        servo_type = "XL330-M077-T"
        speed_limit = torque_limit = 200
        
        try:
            self.lift_servo = Servo(gripper.port_handler, gripper.packet_handler, gripper.protocol, id, led,
                                            torque_limit, speed_limit, self.max_value,
                                            self.min_value, servo_type)
            self.lift_servo.enable()
        except (GripperError, DynamixelServoError) as error:
            raise GripperError(f"Gripper#{self.gripper_id}: Failed to initialise lift servo") from error

    # overriding method
    def _reset(self):
        self.gripper.home()
        # self._wiggle_lift()
        self._grab_cube()
    
    def _lift_up(self):
        self.lift_servo.move(self.max_value,timeout=1)

    def _lift_down(self):
        self.lift_servo.move(self.min_value,timeout=1)

    def _grab_cube(self):
        self._lift_up()
        # check cube is above line?
        self.gripper.move([512,512,362,662])
        self._lift_down()

    def _wiggle_lift(self):
        self.lift_servo.move(2000,timeout=1)
        self._lift_down()
        self.lift_servo.move(1000,timeout=1)
        self._lift_down()

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
    
    def _render_environment(self, state, environment_state):
        # Get base rendering of the two-finger environment translate
        image = super()._render_environment(state, environment_state)

        # Add bottom line for suspended translation task
        bounds_color = (0, 0, 255)

        _, goal_line_pixel_y = utils.position_to_pixel(
            [0,self.bottom_line], self.reference_position, self.camera.camera_matrix
        )

        # goal_line_pixel_y += abs(self.reference_position[1])

        cv2.line(image, (0, int(goal_line_pixel_y)), (640, int(goal_line_pixel_y)), bounds_color, 2)

        return image


    #overriding method
    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False

        reward = 0

        target_goal = current_environment_info["goal"]

        # This now converts the poses with respect to reference marker
        object_previous = self._pose_to_state(previous_environment_info["poses"]["object"])
        object_current = self._pose_to_state(current_environment_info["poses"]["object"])
        logging.debug(f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}")
 
        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)
        
        logging.debug(f"Distance to Goal: {goal_distance_after}")

        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 60
        elif goal_distance_after > self.goal_range or object_current[1] >= self.bottom_line:
            reward = 0
        else:
            reward = round((-goal_distance_after+self.goal_range),2)
        
        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward}"
        )

        print(object_current, reward)

        return reward, done


    def reboot(self):
        logging.info("Rebooting Gripper")
        self.gripper.reboot()
        logging.info("Rebooting Lift Servo")
        self._init_lift_servo(self.gripper)


