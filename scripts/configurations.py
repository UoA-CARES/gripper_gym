from typing import Optional
from cares_reinforcement_learning.util import configurations as cares_cfg

class GripperEnvironmentConfig(cares_cfg.SubscriptableClass):
    task: str

    camera_id: int
    blindable: bool
    observation_type: int
    
    goal_selection_method: int

    marker_size: Optional[int] = 18
    noise_tolerance: Optional[int] = 5

    camera_matrix: str
    camera_distortion: str

    min_noise: float = 0.01
    noise_tolerance: float = 0.9999
    noise_scale: float = 0.1

    episode_horizon: int = 50
    step_length: float = 0.2

    is_debug = False

# TODO find a better name for 'object'
class ObjectConfig(cares_cfg.SubscriptableClass):
    object_type: str
    object_observation_mode: str
    object_marker_id: int

    device_name: str
    baudrate: Optional[int] = 115200
