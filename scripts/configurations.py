from pathlib import Path
file_path = Path(__file__).parent.resolve()

from pydantic import BaseModel
from typing import List, Optional

class LearningConfig(BaseModel):
    seed: int
    batch_size: int
    buffer_capacity: int
    episode_horizont: int

    G: int
    plot_freq: Optional[int] = 10

    max_steps_exploration: int
    max_steps_training: int

    actor_lr: Optional[float]
    critic_lr: float
    gamma: float
    tau: float

class EnvironmentConfig(BaseModel):
    env_type: int

    camera_id: int
    object_type: int
    observation_type: int
    goal_selection_method: int

    noise_tolerance: Optional[int] = 5
    marker_size: Optional[int] = 18

    camera_matrix: Optional[str] = f"{file_path}/config/camera_matrix.txt"
    camera_distortion: Optional[str] = f"{file_path}/config/camera_distortion.txt"

class GripperConfig(BaseModel):
    gripper_type: int
    gripper_id: int
    device_name: str
    baudrate: int
    torque_limit: int
    speed_limit: int
    num_motors: int
    min_values: List[int]
    max_values: List[int]
    home_pose: List[int]
    actuated_target: bool
