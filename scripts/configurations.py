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
    action_type: str
    goal_selection_method: int

    noise_tolerance: Optional[int] = 5
    marker_size: Optional[int] = 18

    camera_matrix: str
    camera_distortion: str

class GripperConfig(BaseModel):
    gripper_type: int
    gripper_id: int
    device_name: str
    baudrate: int
    torque_limit: int
    speed_limit: int
    velocity_min: int
    velocity_max: int
    num_motors: int
    min_values: List[int]
    max_values: List[int]
    home_sequence: List[List[int]]
    actuated_target: bool
