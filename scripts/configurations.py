from pathlib import Path
file_path = Path(__file__).parent.resolve()

from pydantic import BaseModel
from typing import List, Optional

class GripperConfig(BaseModel):
    gripper_type: int
    gripper_id: int
    device_name: str
    baudrate: int
    torque_limit: int
    speed_limit: int
    num_motors: int
    min_value: List[int]
    max_value: List[int]
    home_pose: List[int]
    actuated_target: bool

class EnvironmentConfig(BaseModel):
    # environment
    camera_id: int
    marker_id: int
    marker_size: int
    camera_matrix: Optional[str] = f"{file_path}/config/camera_matrix.txt"
    camera_distortion: Optional[str] = f"{file_path}/config/camera_distortion.txt"

    # gripper
    gripper_config: GripperConfig

class LearningConfig(BaseModel):
    # environment
    seed: int
    batch_size: int
    buffer_capacity: int
    episode_num: int
    action_num: int

    # gripper
    env_config: EnvironmentConfig