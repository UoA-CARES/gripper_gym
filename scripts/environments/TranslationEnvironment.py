from environments.Environment import Environment
import logging
import numpy as np
import math
from pathlib import Path
import random
file_path = Path(__file__).parent.resolve()
from configurations import GripperEnvironmentConfig, ObjectConfig
from cares_lib.dynamixel.gripper_configuration import GripperConfig


class TranslationEnvironment(Environment):
    def __init__(self, env_config : GripperEnvironmentConfig, gripper_config : GripperConfig, object_config: ObjectConfig):
        super().__init__(env_config, gripper_config, object_config)
        self.goal_state = self.get_object_state()

        # Setting max and min values for the target goals

        self.gripper.move(self.gripper.max_values)

        # Just the x and y position of the box
        self.goal_max = self.get_object_state()[:2]

        self.gripper.move(self.gripper.min_values)
        self.goal_min = self.get_object_state()[:2]

        print(f'Goal Max: {self.goal_max}')
        print(f'Goal Min: {self.goal_min}')

   # overriding method
    def choose_goal(self):
        x1, y1 = self.goal_max
        x2, y2 = self.goal_min

        # Distance away from the centre
        n = self.noise_tolerance * 2

        # Calculate the center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # Calculate the direction vector along the line
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize the direction vector
        length = (dx**2 + dy**2)**0.5


        dx /= length
        dy /= length
        # Generate a random point along the line segment
        random_point = (random.uniform(0, 1) * length, random.uniform(0, 1) * length)
        # Ensure it is at least 'n' units away from the center point
        while (random_point[0] - length / 2) ** 2 + (random_point[1] - length / 2) ** 2 < n ** 2:
            random_point = (random.uniform(0, 1) * length, random.uniform(0, 1) * length)
        # Calculate the final point on the line segment
        final_x = center_x + random_point[0] * dx
        final_y = center_y + random_point[1] * dy

        print(f'New Goal: {final_x}, {final_y}')
        return [final_x, final_y]

    def reward_function(self, target_goal, goal_before, goal_after):
        if goal_before is None:
            logging.debug("Start Marker Pose is None")
            return 0, True

        if goal_after is None:
            logging.debug("Final Marker Pose is None")
            return 0, True
        done = False
        
        reward = 0

        goal_distance_before = math.dist(target_goal[:2], goal_before[:2])
        goal_distance_after = math.dist(target_goal[:2], goal_after[:2])

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

        logging.info(f"Reward: {reward}, Goal after: {goal_after[:2]}")

        return reward, done
    
    def ep_final_distance(self):
        return np.linalg.norm(np.array(self.goal_state) - np.array(self.get_object_state()[0:2]))
    
    def add_goal(self, state):
        state.append(self.goal_state[0])
        state.append(self.goal_state[1])
        return state
    
    def env_render(self, done=False, step=1, episode=1, mode="Exploration"):
        image = self.camera.get_frame()
        color = (0, 255, 0)
        if done:
            color = (0, 0, 255)

        target = (int(self.goal_pixel[0]), int(self.goal_pixel[1]))
        text_in_target = (
            int(self.goal_pixel[0]) - 15, int(self.goal_pixel[1]) + 3)
        cv2.circle(image, target, 18, color, -1)
        cv2.putText(image, 'Target', text_in_target,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Episode : {str(episode)}', (30, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Steps : {str(step)}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Success Counter : {str(self.counter_success)}', (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, f'Stage : {mode}', (900, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("State Image", image)
        cv2.waitKey(10)

    

