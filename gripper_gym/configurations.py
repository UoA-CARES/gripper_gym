from typing import Optional
from cares_reinforcement_learning.util import configurations as cares_cfg


class GripperEnvironmentConfig(cares_cfg.SubscriptableClass):
    domain: str
    task: str

    camera_id: int
    camera_matrix: str
    camera_distortion: str
    is_inverted: Optional[bool] = False

    # actions per episode
    episode_horizon: Optional[int] = 50

    # Time steps (secs) between action updates in velocity mode
    step_time_period: Optional[float] = 0.2  # secs

    # Aruco or STAG Marker size in mm
    marker_size: Optional[int] = 18  # mm

    # Aruco Marker ID for the object
    object_marker_id: Optional[int] = 7
    reference_marker_id: Optional[int] = 1
    cube_ids: Optional[list] = [1,2,3,4,5,6]

    # Tolerance in position error for object being at goal
    noise_tolerance: Optional[int] = 5  # mm or degrees

    # Rotation Environment specific
    object_device_name: Optional[str] = "/dev/ttyUSB0"
    object_baudrate: Optional[int] = 115200
    # TODO make a string enum
    goal_selection_method: Optional[int] = 0

    # Translation Environment specific
    elevator_device_name: Optional[str] = "/dev/ttyUSB0"
    elevator_baudrate: Optional[int] = 1000000
    elevator_servo_id: Optional[int] = 10
    elevator_limits: Optional[list] = [0,0] # [MAX,MIN]

    is_debug = False

    # For when ssh to train, display can be turned off 
    display: Optional[bool] = True

    reward_function: Optional[str] = "staged"
