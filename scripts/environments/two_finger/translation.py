import logging
import math
from random import randrange
from random import choice
import time

import cv2
import tools.utils as utils
from configurations import GripperEnvironmentConfig
from environments.two_finger.two_finger import TwoFingerTask

from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_lib.dynamixel.Servo import Servo
import dynamixel_sdk as dxl


class TwoFingerTranslation(TwoFingerTask):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        self.noise_tolerance = env_config.noise_tolerance

        # These bounds are respective to the reference marker in Environment
        self.goal_min = [-30.0, 70.0]
        self.goal_max = [120.0, 120.0]

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

        # Simple goal implementation
        # x_selection = [-20, 110]
        # goal_x = choice(x_selection)
        # goal_y = 95

        return [goal_x, goal_y]

    # overriding method
    def _environment_info_to_state(self, environment_info):
        state = []

        # Servo Angles - Steps
        #state += environment_info["gripper"]["positions"]

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

        # This now converts the poses with respect to reference marker
        object_previous = self._pose_to_state(previous_environment_info["poses"]["object"])
        object_current = self._pose_to_state(current_environment_info["poses"]["object"])
        logging.debug(f"Prev object: {object_previous}  Current object: {object_current} Target: {target_goal}")

        goal_distance_before = math.dist(target_goal, object_previous)
        goal_distance_after = math.dist(target_goal, object_current)

        goal_progress = goal_distance_before - goal_distance_after
        
        logging.debug(f"Distance to Goal: {goal_distance_after}")

        delta = 10* (goal_progress/goal_distance_before)
        # For Translation. noise_tolerance is 15, it would affect the performance to some extent.
        if goal_distance_after <= self.noise_tolerance:
            logging.info("----------Reached the Goal!----------")
            reward = 50
        elif goal_progress > 0:
            reward = round(delta, 2)
        else:
            reward = round(max(-10, delta), 2)

        
        logging.debug(
            f"Object Pose: {object_current} Goal Pose: {target_goal} Reward: {reward} Raw-reward: {(goal_progress/goal_distance_before)}"
        )

        return reward, done

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
        bounds_color = (0, 255, 0)
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

        return image


class TwoFingerTranslationFlat(TwoFingerTranslation):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)
        self.elevator_device_name = env_config.elevator_device_name
        self.elevator_baudrate = env_config.elevator_baudrate
        self.elevator_servo_id = env_config.elevator_servo_id
        #TODO add instantiation of the elevator elevator servo here 

    def init_elevator(self):
        self.elevator_port_handler = dxl.PortHandler(self.elevator_device_name)
        self.elevator_packet_handler = dxl.PacketHandler(2)
        self.elevator = Servo(self.elevator_port_handler, self.elevator_packet_handler, 2, self.elevator_servo_id, 1, 200, 200, 1023, 0, model="XL-320")

        if not self.elevator_port_handler.openPort():
            error_message = f"Failed to open port {self.elevator_device_name}"
            logging.error(error_message)    
            raise IOError(error_message)
        logging.info(f"Succeeded to open port {self.elevator_device_name}")

        if not self.elevator_port_handler.setBaudRate(self.elevator_baudrate):
            error_message = f"Failed to change the baudrate to {self.elevator_baudrate}"
            logging.error(error_message)
            raise IOError(error_message)
        logging.info(f"Succeeded to change the baudrate to {self.elevator_baudrate}")
        self.elevator.enable() 

    # overriding method
    def _reset(self):
        self.gripper.wiggle_home()
        # self.init_elevator()

        # # check if object between fingertips
        # def is_object_between():
        #     try:
        #         marker_poses = self._get_marker_poses()
        #     except:
        #         return False
        #     logging.info(marker_poses)
        #     # object_pose = marker_poses[6]
        #     return marker_poses[6]["position"][0] <= object_state[0] <= marker_poses[5]["position"][0]
        

        # # reset until in default position
        # #TODO implement object centred check
        # while not (self.gripper.is_home()):
        #     # reset gripper
        #     self.gripper.move([312, 712, 512, 512])
        #     self.gripper.disable_torque()
        #     time.sleep(1)
        #     self.elevator.move(0)
        #     time.sleep(5)
        #     self.elevator.move(1023)
        #     time.sleep(1)
        #     self.gripper.home()
        #     while(1):
        #         time.sleep(1)


class TwoFingerTranslationSuspended(TwoFingerTranslation):
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        gripper_config: GripperConfig,
    ):
        super().__init__(env_config, gripper_config)

        # TODO add instatiation of the lift servo etc here

    # overriding method
    def _reset(self):
        self.gripper.wiggle_home()

        # Execute code to reset the elevator etc
