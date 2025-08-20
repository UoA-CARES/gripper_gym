import cv2
import numpy as np
import pydantic
from cares_lib.dynamixel.gripper_configuration import GripperConfig


def load_gripper_config(config_path: str) -> GripperConfig:
    try:
        return pydantic.parse_file_as(path=config_path, type_=GripperConfig)
    except FileNotFoundError as e:
        error_msg = f"Gripper config file not found: {config_path}"
        raise FileNotFoundError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to load gripper config from {config_path}: {e}"
        raise ValueError(error_msg) from e


def draw_circle(
    image,
    position_mm: list[float],
    size_mm: float,
    camera_matrix,
    color: tuple[int, int, int],
    reference_position_mm: list[float] = [0, 0, 0],
):
    pixel_location = position_to_pixel(
        position_mm,
        reference_position_mm,
        camera_matrix,
    )

    noise_tolerance_pixels = mm_to_pixels(
        size_mm, reference_position_mm[2], camera_matrix
    )

    # Circle size now reflects the "Close enough" to goal tolerance
    cv2.circle(image, pixel_location, int(noise_tolerance_pixels), color, -1)
    return image, pixel_location


def mm_to_pixels(size_mm, distance_mm, camera_matrix):
    fx = camera_matrix[0, 0]
    return int((size_mm * fx) / distance_mm)


def position_to_pixel(position, reference_position, camera_matrix):
    # pixel_n = f * N / Z + c_n
    pixel_x = (
        camera_matrix[0, 0]
        * (position[0] + reference_position[0])
        / reference_position[2]
        + camera_matrix[0, 2]
    )
    pixel_y = (
        camera_matrix[1, 1]
        * (position[1] + reference_position[1])
        / reference_position[2]
        + camera_matrix[1, 2]
    )
    return int(pixel_x), int(pixel_y)


def angular_difference(angle_a: float, angle_b: float) -> float:
    """
    Compute the minimum absolute angular difference between two angles in degrees.
    Works for any real inputs (not just [0, 360)).

    Args:
        angle_a (float): First angle in degrees.
        angle_b (float): Second angle in degrees.

    Returns:
        float: Minimum angular difference in [0, 180].
    """
    diff = abs((angle_a - angle_b) % 360)
    return min(diff, 360 - diff)


def get_cube_pose(
    marker_poses: dict,
    cube_ids: tuple[int, int, int, int, int, int],
    cube_size: float = 50,
) -> dict | None:
    """
    Calculate the center point of a cube base on the detected markers.
    Args:
        marker_poses (dict): A dictionary containing the poses of the detected ArUco markers.
        cube_ids (tuple[int, int, int, int, int, int]): IDs of the cube markers.
    Returns:
        dict: A dictionary containing the position and orientation of the cube.
    """
    detected_ids = [ids for ids in marker_poses]

    cube_marker_ids = [id for id in cube_ids if id in detected_ids]

    if len(cube_marker_ids) == 0:
        # If no cube marker detected then return a default pose assuming the cube is not visible
        return None

    # Calculate the cube centers for the marker IDs present in both cube_ids and detected_ids
    cube_centers = np.array(
        [calculate_cube_center(marker_poses[id], cube_size) for id in cube_marker_ids]
    )

    # Calculate the final cube center by averaging
    cube_center = np.mean(cube_centers, axis=0)

    return {"position": cube_center, "orientation": [0.0, 0.0, 0.0]}


def calculate_cube_center(marker_pose: dict, cube_size: float) -> np.ndarray:
    """
    Calculate the center point of a cube given the position and orientation of one face.
    Args:
        marker_pose (dict): A dictionary containing the position and orientation of the marker.
        cube_size (int): The size of the cube.
    Returns:
        numpy.ndarray: A 1D array of length 3 representing the x, y, z coordinates of the center of the cube.
    """

    marker_position = np.array(marker_pose["position"])
    r_vec = np.array(marker_pose["orientation"]["r_vec"])

    # Calculate the rotation matrix from the Rodrigues vector
    rotation_matrix, _ = cv2.Rodrigues(r_vec)

    # Calculate the offset from the face center to the cube center
    offset = np.dot(rotation_matrix, np.array([0, 0, cube_size / 2]))

    # Calculate the cube center
    cube_center = marker_position - offset

    return cube_center
