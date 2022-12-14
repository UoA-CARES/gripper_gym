import cv2
import math
import numpy as np

'''
Camera Class
--> init
--> get frame
--> find aruco marker
--> figure out the angle of the marker

'''

class Camera:
    def __init___(self, camera_id):
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   #i dont know how important this is 
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  #but need to get datasets to see what camera default is 

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50) #aruco dictionary
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.markerSize = 18 #mm

        full_path_camera_matrix = "C:\Users\bethc\Documents\git\Gripper-Code\utilities"

        self.vision_flag_status = False
        self.matrix = np.loadtxt((full_path_camera_matrix + "/matrix.txt"))
        self.distortion = np.loadtxt((full_path_camera_matrix + "/distortion.txt"))

    def get_frame(self): 
        ret, frame = self.camera.read()
        if ret: 
            return frame 
        else: print("Error: No frame returned")