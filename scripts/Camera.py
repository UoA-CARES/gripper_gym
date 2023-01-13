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

    def __init__(self, camera_id=0):  # 0 is usb camera, 1 is laptop camera (when the usb camera is plugged in )

        # connecting to the camera takes the longest
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise Exception("Could not open video device")

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)  # aruco dictionary
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_size = 18  # mm
        #TODO: make this relative
        self.camera_matrix = np.loadtxt(("/home/anyone/gripperCode/Gripper-Code/scripts/config/camera_matrix.txt"))
        self.camera_distortion = np.loadtxt(("/home/anyone/gripperCode/Gripper-Code/scripts/config/camera_distortion.txt"))

    def get_frame(self):  
        returned, frame = self.camera.read()
        if returned:
            return frame
        print("Error: No frame returned")
        return None

