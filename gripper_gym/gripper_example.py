import cv2
import os
import logging

logging.basicConfig(level=logging.INFO)
import pydantic
import numpy as np
import time

from pathlib import Path

file_path = Path(__file__).parent.resolve()

import time

from cares_lib.dynamixel.Gripper import Gripper, GripperError
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_lib.dynamixel.Servo import Servo, DynamixelServoError, OperatingMode

import dynamixel_sdk as dxl

import gripper_gym.tools.utils as utils
from cares_lib.vision.Camera import Camera
from cares_lib.vision.STagDetector import STagDetector


def get_marker_poses(
    must_see_ids: list[int], camera, marker_detector, is_inverted
) -> dict[int, dict]:
    while True:
        logging.debug(f"Attempting to Detect markers: {must_see_ids}")
        frame = (
            cv2.rotate(camera.get_frame(), cv2.ROTATE_180)
            if is_inverted
            else camera.get_frame()
        )

        cv2.imshow("Camera Frame", frame)
        cv2.waitKey(10)

        marker_poses = marker_detector.get_marker_poses(
            frame,
            camera.camera_matrix,
            camera.camera_distortion,
        )

        # This will check that all the markers are detected correctly
        if all(ids in marker_poses for ids in must_see_ids):
            break

    return marker_poses


# Example of how to use Gripper
def main():
    gripper_id = 1

    # camera_name = f"/dev/camera{gripper_id}"
    camera_name = f"/dev/video0"

    calibration_path = os.path.expanduser(f"~/gripper_configs/{gripper_id}")
    camera_matrix_path = os.path.join(calibration_path, "camera_matrix.txt")
    camera_distortion_path = os.path.join(calibration_path, "camera_distortion.txt")

    camera = Camera(camera_name, camera_matrix_path, camera_distortion_path)

    stag_detector = STagDetector(marker_size=32.0, library_hd=11)

    while True:
        marker_poses = get_marker_poses(
            must_see_ids=[],
            camera=camera,
            marker_detector=stag_detector,
            is_inverted=False,
        )

        cube_pose = utils.get_cube_pose(
            marker_poses=marker_poses,
            cube_ids=[1, 2, 3, 4, 5, 6],
            cube_size=32.0,
        )

        if cube_pose is not None:
            degrees = np.array(cube_pose["orientation"])

            print(f"Cube Position: {cube_pose['position']} Orientation: {degrees}")

            time.sleep(0.5)


def camera_test():
    # camera = cv2.VideoCapture("/dev/video4")
    camera = cv2.VideoCapture("/dev/video0")
    if not camera.isOpened():
        raise IOError("Cannot open camera")

    while True:
        for _ in range(0, 5):
            returned, frame = camera.read()

        cv2.imshow("Camera Test", frame)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
    # camera_test()
