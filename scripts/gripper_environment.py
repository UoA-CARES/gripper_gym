import numpy as np
#import dynamixel_sdk as dxl

from Gripper import Gripper
from Camera import Camera
from cares_lib.vision.ArucoDetector import ArucoDetector

class Environment():


#init
    def __init__(self):
        self.gripper = Gripper()
        self.camera = Camera()
        self.aruco_detector = ArucoDetector(marker_size=18)

#reset
    def reset(self):
        self.gripper.home()


#reward function 
    def reward_function(self, target_angle, valve_angle_previous, valve_angle_after, action_taken, terminated):
            delta_changes = np.abs(target_angle - valve_angle_previous) - np.abs(target_angle - valve_angle_after)
            angle_difference = np.abs(target_angle - valve_angle_after)

            print(f"target_angle: {target_angle}, delta changes: {delta_changes}, angle difference: {angle_difference}")

            reward = (-angle_difference+360)+(-(10*action_taken)+150)

            if valve_angle_previous == -1 or valve_angle_after == -1:
                reward = 0

            elif terminated:
                reward += -1000

            elif angle_difference < 10:
                reward += 10000

            else:
                done = False

            return reward

#step
    def step(self, action, target_angle, action_taken):

        #TODO: change the 0 in all the marker pose indexing to an aruco id variable
        #TODO: get check to do something when the aruco marker can't be found

        Done = False
        terminated = False

        while not Done:
            frame = self.camera.get_frame()
            start_marker_pose = self.aruco_detector.get_marker_poses(
                frame, self.camera.camera_matrix, self.camera.camera_distortion)
            
            if len(start_marker_pose) == 0:
                start_marker_pose = -1
            else:
                start_marker_pose = start_marker_pose[0][1][2]
            #print(f"start_marker_pose: {start_marker_pose}")
            
            # Take Action (maybe this needs to be in its own while loop?)
            state, terminated = self.gripper.move(action)
            #print(f"state: {state}")

            Done = True #need this to tell that the action is complete

        Done = False #need to reset so that i can tell the 
        # Measure State of aruco marker
        if terminated: 
            Done = True

        final_aruco_position = self.aruco_detector.get_marker_poses(frame, self.camera.camera_matrix, self.camera.camera_distortion)
        if len(final_aruco_position) == 0:
            final_marker_pose = -1
        else:
            final_marker_pose = final_aruco_position[0][1][2]
        #print(f"final_marker_pose: {final_marker_pose}")

        state.append(final_marker_pose)
        #print(f"state being returned {state}")
        
        # Calculate Reward, figure out how to index marker_pose
        reward = self.reward_function(target_angle, start_marker_pose, final_marker_pose, action_taken, terminated)

        goal_reached = 0

        if (target_angle-10)<final_marker_pose<(target_angle+10):
            terminated = False
            Done = True #not to sure about this but keeping it for now
            goal_reached += 1
            print(f"goal reached = {goal_reached}")
        
        return state, reward, terminated, Done
            
