from cares_lib.vision.ArucoDetector import ArucoDetector
from cares_lib.vision.Camera import Camera
from Objects import MagnetObject
from configurations import ObjectConfig
import time
from scipy.stats import trim_mean
import logging

logging.basicConfig(level=logging.INFO)

def rotation_min_difference(a, b):
    return min(abs(a - b), (360+min(a, b) - max(a, b)))

def get_marker_pose(aruco_detector, camera):
    frame = camera.get_frame()
    marker_poses = aruco_detector.get_marker_poses(frame, camera.camera_matrix, camera.camera_distortion, display=False)
    if 4 in marker_poses:
        aruco_yaw = marker_poses[4]["orientation"][2]
        return aruco_yaw
    return None
        

def main():
    camera_id = 2
    camera_matrix = "/home/anyone/gripper_local_storage/config/camera_matrix.txt"
    camera_distortion = "/home/anyone/gripper_local_storage/config/camera_distortion.txt"
    camera  = Camera(camera_id, camera_matrix, camera_distortion)

    aruco_detector = ArucoDetector(marker_size=18)

    aruco_yaws = []
    for i in range(0, 10):
        aruco_yaws.append(get_marker_pose(aruco_detector, camera))
    aruco_yaw = trim_mean(aruco_yaws, 0.1)

    object_config = ObjectConfig(target_type="magnet", device_name = "/dev/ttyACM0", baudrate = 115200)
    target = MagnetObject(object_config, aruco_yaw)

    while True:
        magnet_yaw = target.get_yaw()
        while magnet_yaw is None:
            magnet_yaw = target.get_yaw()

        aruco_yaw = get_marker_pose(aruco_detector, camera)
        while aruco_yaw is None:
            aruco_yaw = get_marker_pose(aruco_detector, camera)
        print(f"Magnet {magnet_yaw} Aruco {aruco_yaw}. Diff: {rotation_min_difference(magnet_yaw, aruco_yaw)}")

        # time.sleep(0.2)

# def main():
#     from serial import Serial
#     from time import sleep
#     serial = Serial("/dev/ttyACM0", 115200)
#     # while not serial.is_open:
#     #     print("MagnetObject: Waiting for serial port to open...")
#     #     sleep(1)
#     #     pass
#     sleep(1)
#     print(serial.write(b"0,\n"))
#     print(serial.read_until(b'\n').decode())

if __name__ == '__main__':
    main()