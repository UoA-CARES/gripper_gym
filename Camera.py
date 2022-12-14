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
        
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise Exception("Could not open video device")
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   #i dont know how important this is 
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  #but need to get datasets to see what camera default is 
        
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50) #aruco dictionary
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.markerSize = 18 #mm
      
        self.matrix = np.loadtxt(("C:\\Users\\bethc\\Documents\\git\\Gripper-Code\\utilities\\matrix.txt"))
        self.distortion = np.loadtxt(("C:\\Users\\bethc\\Documents\\git\\Gripper-Code\\utilities\\distortion.txt"))
        self.vision_flag_status = False
    
    def get_frame(self):
    
        ret, frame = self.camera.read()
        if ret: 
            cv2.imshow("frame", frame)
            cv2.waitKey(0)
        else: print("Error: No frame returned")
        