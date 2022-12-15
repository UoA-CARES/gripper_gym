import cv2
import math
import numpy as np
import os
import time

'''
Camera Class
--> init
--> get frame
--> find aruco marker
--> figure out the angle of the marker

'''

class Camera(object):
    def __init__(self, camera_id=0):    #0 is usb camera, 1 is laptop camera
        
        #connecting to the camera takes the longest
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise Exception("Could not open video device") 
    
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50) #aruco dictionary
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.markerSize = 18 #mm
        self.matrix = np.loadtxt(("C:\\Users\\bethc\\Documents\\git\\Gripper-Code\\utilities\\matrix.txt"))
        self.distortion = np.loadtxt(("C:\\Users\\bethc\\Documents\\git\\Gripper-Code\\utilities\\distortion.txt"))
        self.vision_flag_status = False

    def get_frame(self):
    
        ret, frame = self.camera.read()
        if ret: 
            # cv2.imshow("frame", frame)  #this is temporary but useful for debugging
            # cv2.waitKey(0)
            self.vision_flag_status = True
            return frame
        else: print("Error: No frame returned")
        
    def detect_display(self):

        #lower_blue = np.array([110,50,50], datatype = "unit8")  #this is bgr thanks copilot
        #upper_blue = np.array([255,255,130], datatype ="uint8")  #this is bgr
            
            frame = self.get_frame()
            #if arcuo marker is detected, draw it
            (corners, ids, rejectedImgPoints) = cv2.aruco.detectMarkers(frame, self.arucoDict, parameters=self.arucoParams)
            if len(corners) > 0:
                cv2.putText(frame, f'Cylinder Angle : {self.get_marker_pose(180)[0]}', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                #detecting the aruco marker
                cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 0, 255))
                cv2.imshow("frame", frame)
                cv2.waitKey(5)

            #if the arcuo marker is not detected, keep the while loop running but display a message sying it is not detected
            elif len(corners) == 0:
                cv2.putText(frame, "No Aruco Marker Detected", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("frame", frame)
                cv2.waitKey(5)
                



    def get_marker_pose(self, goal_angle):
        frame = self.get_frame()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.arucoDict, parameters=self.arucoParams)
        r_vec, t_vec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.markerSize, self.matrix, self.distortion)

        valve_location = t_vec[0][0][0:2]
        
        valve_angle = np.array([self.get_angle(r_vec[0][0])])
        print(valve_angle,  valve_location)

        goal_angle = np.array([goal_angle])

        state_space = (valve_location, valve_angle, goal_angle)

        self.vision_flag_status = True
        return valve_angle, state_space, frame, self.vision_flag_status

    def is_close(self, x, y, rtol=1.e-5, atol=1.e-8):
        return abs(x - y) <= atol + rtol * abs(y)   #this is a tolerance thingy i think

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
        psi, theta, phi = self.calculate_euler_angles(r_matrix)  #roll, pitch, yaw
        phi = math.degrees(phi)
        if phi < 0:         #figure out how to generalise this
            phi += 360
        return phi
