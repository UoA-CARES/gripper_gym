import numpy as np
import dynamixel_sdk as dxl

from Gripper import Gripper
from Camera import Camera
from cares_lib.vision.ArucoDetector import ArucoDetector

# TODO: sort this all out um yep make this work better with the testing loop

# def reward_function(target_angle, valve_angle_previous, valve_angle_after):
#         delta_changes = np.abs(
#             target_angle - valve_angle_previous) - np.abs(target_angle - valve_angle_after)
#         angle_difference = np.abs(target_angle - valve_angle_after)

#         if -5 <= delta_changes <= 5:
#             # noise or no changes
#             reward_ext = 0
#         else:
#             reward_ext = delta_changes

#         if angle_difference <= 5:

#             reward_ext = reward_ext + 100
#             print(f"reward: {reward_ext}")
#             done = True
#         else:
#             done = False

#         return reward_ext, done, angle_difference

# def move(self, action, target_angle):  # the action input should be in steps
#         done = False

#         # loop through the actions array and send the commands to the motors
#         if action.shape[0] != self.num_motors:
#             print("Error: action array is not the correct length")
#             # quit()

#         # print(action)
#         action = self.verify_actions(action)

#         for servo in self.servos:

#             servo = self.servos[servo]
#             # add parameters to the groupSyncWrite
#             print(action)
#             self.group_sync_write.addParam(servo.motor_id, [dxl.DXL_LOBYTE(
#                 action[servo.motor_id-1]), dxl.DXL_HIBYTE(action[servo.motor_id-1])])

#         pre_valve_angle = self.camera.get_marker_pose(0)[0]
#         # transmit the packet
#         dxl_comm_result = self.group_sync_write.txPacket()
#         if dxl_comm_result == dxl.COMM_SUCCESS:
#             print("group_sync_write Succeeded")

#         self.group_sync_write.clearParam()

#         # using moving flag to check if the motors have reached their goal position
#         while self.gripper_moving_check():
#             self.camera.detect_display()
#             self.all_current_positions()

#         post_valve_angle = self.camera.get_marker_pose(0)[0]
#         reward = self.reward_function(target_angle, pre_valve_angle, post_valve_angle)

#         if post_valve_angle == target_angle:
#             done = True

#         # once its done, append the valve position to the list and return? at least return current position
#         return self.all_current_positions(), reward[0], done


def main():

    # Set Camera Up

    # Set Gripper Up

    # Setup Arcuo Detector Up 

    camera = Camera()
    aruco_detector = ArucoDetector(marker_size=18)

    while True:
        frame = camera.get_frame()
        marker_poses = aruco_detector.get_marker_poses(
            frame, camera.camera_matrix, camera.camera_distortion)
        print(marker_poses)

    # while learning
        # Take Action
        # Measure State of aruco marker
        # Calculate Reward
        # Learn

if __name__ == "__main__":
    main()
