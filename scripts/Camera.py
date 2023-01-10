import cv2
import math
import numpy as np

'''
WORK IN PROGRESS
Camera Class

#TODO: refactor this code so the methods make a bit more sense to use with each other
'''

from pathlib import Path
home = str(Path.home())

class Camera(object):

    def __init__(self, camera_id=0):  # 0 is usb camera, 1 is laptop camera

        # connecting to the camera takes the longest
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise Exception("Could not open video device")

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)  # aruco dictionary
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.markerSize = 18  # mm
        self.matrix = np.loadtxt(
            (home+"Documents\\git\\rlstuff\\Gripper-Code\\scripts\\utilities\\matrix.txt"))
        self.distortion = np.loadtxt(
            (home+"Documents\\git\\rlstuff\\Gripper-Code\\scripts\\utilities\\distortion.txt"))
        self.vision_flag_status = False


    def get_frame(self):  

        ret, frame = self.camera.read()
        if ret:
            # cv2.imshow("frame", frame)  #this is temporary but useful for debugging
            # cv2.waitKey(0)
            self.vision_flag_status = True
            return frame
        else:
            print("Error: No frame returned")


    def detect_display(self):

        frame = self.get_frame()
        # if arcuo marker is detected, draw it
        (corners, ids, rejectedImgPoints) = cv2.aruco.detectMarkers(
            frame, self.arucoDict, parameters=self.arucoParams)

        if len(corners) > 0:
            # ok still don't know why this is throwing an error
            cv2.putText(frame, f'Cylinder Angle : {self.get_marker_pose(180)[0]}', (
                60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            # detecting the aruco marker
            cv2.aruco.drawDetectedMarkers(
                frame, corners, ids, borderColor=(0, 0, 255))
            cv2.imshow("frame", frame)
            cv2.waitKey(5)

        # if the arcuo marker is not detected, keep the while loop running but display a message sying it is not detected
        elif len(corners) == 0:
            cv2.putText(frame, "No Aruco Marker Detected", (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("frame", frame)
            cv2.waitKey(5)


    def get_marker_pose(self, goal_angle):

        frame = self.get_frame()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            frame, self.arucoDict, parameters=self.arucoParams)
        # check if the aruco marker is detected --> basically make this function and the one above better
        if corners:
            r_vec, t_vec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.markerSize, self.matrix, self.distortion)

            valve_location = t_vec[0][0][0:2]

            valve_angle = self.get_angle(r_vec[0][0])
            #print(valve_angle,  valve_location)

            goal_angle = np.array([goal_angle])

            self.vision_flag_status = True
        else:
            valve_angle = -1 #set to previous angle somehow
            valve_location = -1

        return valve_angle, frame, self.vision_flag_status


    def is_close(self, x, y, rtol=1.e-5, atol=1.e-8):
        # this is a tolerance thingy i think
        return abs(x - y) <= atol + rtol * abs(y)


    def calculate_euler_angles(self, R):

        phi = 0.0
        if self.is_close(R[2, 0], -1.0):
            theta = math.pi / 2.0
            psi = math.atan2(R[0, 1], R[0, 2])
        elif self.is_close(R[2, 0], 1.0):
            theta = -math.pi / 2.0
            psi = math.atan2(-R[0, 1], -R[0, 2])
        else:
            theta = -math.asin(R[2, 0])
            cos_theta = math.cos(theta)
            psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
            phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
        return psi, theta, phi


    def get_angle(self, r_vec):
        r_matrix, _ = cv2.Rodrigues(r_vec)
        psi, theta, phi = self.calculate_euler_angles(
            r_matrix)  # roll, pitch, yaw
        phi = math.degrees(phi)
        if phi < 0:  # figure out how to generalise this, make into another function?
            phi += 360
        return phi