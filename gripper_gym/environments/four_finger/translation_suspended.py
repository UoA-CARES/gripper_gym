import logging
import math
import time

import dynamixel_sdk as dxl
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_lib.dynamixel.Servo import Servo
from cares_lib.vision.STagDetector import STagDetector

from gripper_gym.configurations import GripperEnvironmentConfig
from gripper_gym.environments.four_finger.translation import FourFingerTranslation


class FourFingerTranslationSuspended(FourFingerTranslation):
    def __init__(
        self, env_config: GripperEnvironmentConfig, gripper_config: GripperConfig
    ):
        self.env_config = env_config
        self.gripper_config = gripper_config
        self.noise_tolerance = env_config.noise_tolerance
        self.aruco_detector = STagDetector(
            marker_size=env_config.marker_size, library_hd=11
        )
        super().__init__(env_config, gripper_config)
        self.elevator_device_name = env_config.elevator_device_name
        self.elevator_baudrate = env_config.elevator_baudrate
        self.elevator_servo_id = env_config.elevator_servo_id
        self.elevator_min = env_config.elevator_limits[0]  # Lowered Elevator Position
        self.elevator_max = env_config.elevator_limits[1]  # Extended Elevator Position

        self.total_moves = 0
        self.total_lifts = 0

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
            model="XL330-M077-T",
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
        self.elevator.move(self.elevator_min)  # Lower Elevator
        self.gripper.wiggle_home()  # Home Gripper
        # Opening Grasp
        if self.gripper_config.touch:
            time.sleep(1)
            self.tactile_server.baseline_values = self.get_values(self.socket_port)
            print("Baseline values updated", self.tactile_server.baseline_values)
        self.elevator.move(self.elevator_max)  # Raise Elevator
        self.gripper.move(
            [2048, 2250, 2350, 2048, 2250, 2350, 2048, 2250, 2350, 2048, 2250, 2350]
        )  # Grasp Cube
        self.elevator.move(self.elevator_min)

    def _reward_function(self, previous_environment_info, current_environment_info):
        done = False
        reward = 0
        self.goal_reward = 400
        height_threshold = 200
        movement_threshold = 10
        if self.iscubedropped:
            current_height = 230
        else:
            current_height = current_environment_info["poses"]["object"]["position"][2]
        print("Current Height: ", current_height)

        # Get current and previous distance to goal
        target_pose = current_environment_info["goal"]
        target_pose = [
            target_pose[0] + self.reference_position[0],
            target_pose[1] + self.reference_position[1],
        ]
        current_object_pose = current_environment_info["poses"]["object"]["position"][
            0:2
        ]
        previous_object_pose = previous_environment_info["poses"]["object"]["position"][
            0:2
        ]

        # Calculate the distance to the goal
        previous_goal_distance = math.dist(target_pose, previous_object_pose)
        current_goal_distance = math.dist(target_pose, current_object_pose)
        print(
            "Previous Goal Distance: ",
            previous_goal_distance,
            "Current Goal Distance: ",
            current_goal_distance,
        )

        # Touch-based reward oustside of height threshold check
        # if self.step_counter == 5 or current_height < height_threshold:
        if self.touch_config == True:
            num_touch = 0
            touch_threshold = 3
            touch_reward = 20
            print("Getting touch data in reward function")
            print("Max values after step: ", self.tactile_server.max_values)
            # Do reward based on touch sensor values
            for i in range(self.num_sensors):
                delta_touch = (
                    self.tactile_server.max_values[i]
                    - self.tactile_server.baseline_values[i]
                )
                if delta_touch < 0:
                    continue
                if delta_touch < touch_threshold:
                    continue
                else:
                    num_touch += 1
            # Reward based on number of touch sensors triggered
            reward += num_touch * touch_reward
            print(
                "Number of touch sensors triggered: ",
                num_touch,
                "Reward: ",
                num_touch * touch_reward,
            )
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
        # if current_height < height_threshold:
        #     # Stage 1
        #     reward += 100
        #     self.total_lifts += 1

        #     delta = previous_goal_distance - current_goal_distance
        #     print("Delta: ", delta)

        #     if self.total_moves < 50 and self.total_lifts > 5:
        #         #Stage 2
        #         if -movement_threshold < delta < movement_threshold:
        #             # Did not move cube
        #             reward -= 50
        #         else:
        #             # Did move cube
        #             reward += 50
        #             self.total_moves += 1
        #     elif self.total_moves > 50 and self.total_lifts > 5:
        #         #Stage 3
        #         raw_reward = (delta/previous_goal_distance)
        #         print("Raw Reward: ", raw_reward*self.goal_reward)
        #         reward += round((raw_reward*self.goal_reward), 2)
        # else:
        #     # Cube fallen threshold
        #     reward += -100

        print("Moved", self.total_moves, "Lifts", self.total_lifts)
        # Check if the goal is reached
        if (
            current_goal_distance < self.noise_tolerance
            and current_height < height_threshold
        ):
            reward = self.goal_reward
            logging.info(f"Goal Reached!")
        reward = round(reward, 2)
        print(f"Total Reward: ", reward)
        return reward, done
