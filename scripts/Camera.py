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
    def __init__(self, camera_id=0):

        # connecting to the camera takes the longest
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise Exception("Could not open video device")

        #TODO: make this relative and parameters into the camera object
        self.camera_matrix = np.loadtxt((home+"/gripperCode/Gripper-Code/scripts/config/camera_matrix.txt"))
        self.camera_distortion = np.loadtxt((home+"/gripperCode/Gripper-Code/scripts/config/camera_distortion.txt"))

    def get_frame(self):  
        returned, frame = self.camera.read()
        if returned:
            return frame
        print("Error: No frame returned")
        return None

