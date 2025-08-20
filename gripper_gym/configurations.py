from typing import Optional

from pydantic import BaseModel


class SubscriptableClass(BaseModel):
    def __getitem__(self, item):
        return getattr(self, item)


class GripperEnvironmentConfig(SubscriptableClass):
    domain: str
    task: str

    gripper_id: int

    # actions per episode
    episode_horizon: Optional[int] = 50

    # Time steps (secs) between action updates in velocity mode
    step_time_period: Optional[float] = 0.2  # secs

    aruco_detector: str = "STag"  # or "STag"

    # Aruco or STAG Marker size in mm
    marker_size: Optional[int] = 18  # mm

    # Aruco Marker ID for the object
    reference_marker_id: int = 1

    # Tolerance in position error for object being at goal
    noise_tolerance: float  # mm or degrees

    goal_selection_method: Optional[int] = 0

    is_debug = False

    # For when ssh to train, display can be turned off
    display: Optional[bool] = True

    # Camera configuration
    is_inverted: Optional[bool] = False

    # Use touch sensors
    use_touch: Optional[bool] = False


class TranslationConfig(GripperEnvironmentConfig):
    """
    Configuration for the Translation environment.
    Inherits from GripperEnvironmentConfig.
    """

    task: str = "translation"

    goal_min: list[float]  # mm
    goal_max: list[float]  # mm

    noise_tolerance: float = 10.0


#########################################
# Two Finger Environment Configurations #
#########################################


class TwoFingerConfig(GripperEnvironmentConfig):
    """
    Configuration for the Two Finger environment.
    Inherits from GripperEnvironmentConfig.
    """

    domain: str = "two_finger"


class TwoFingerTranslationConfig(TwoFingerConfig):
    """
    Configuration for the Two Finger Translation environment.
    Inherits from TwoFingerConfig.
    """

    goal_min: list[float]  # mm
    goal_max: list[float]  # mm

    noise_tolerance: float = 10.0  # mm

    # Translation Environment specific
    elevator_baudrate: int = 1000000
    elevator_servo_id: int
    elevator_limits: list  # [MAX,MIN]


class TwoFingerFlatConfig(TwoFingerTranslationConfig):
    """
    Configuration for the Two Finger Flat environment.
    Inherits from TwoFingerConfig.
    """

    task: str = "translation"

    goal_min: list[float] = [-40.0, 70.0]  # mm
    goal_max: list[float] = [100.0, 110.0]  # mm

    elevator_servo_id: int = 5
    elevator_limits: list = [3000, 1000]  # [MAX,MIN]

    is_inverted: Optional[bool] = True


class TwoFingerSuspendedConfig(TwoFingerTranslationConfig):
    """
    Configuration for the Two Finger Suspended environment.
    Inherits from TwoFingerConfig.
    """

    task: str = "suspended_translation"

    goal_min: list[float] = [-20.0, 70.0]  # mm
    goal_max: list[float] = [100.0, 105.0]  # mm

    elevator_servo_id: int = 5
    elevator_limits: list = [6000, 1200]  # [MAX,MIN]

    reward_function: str = "staged"


class TwoFingerRotationConfig(TwoFingerConfig):
    """
    Configuration for the Two Finger Rotation environment.
    Inherits from TwoFingerConfig.
    """

    task: str = "rotation"

    noise_tolerance: float = 5.0  # degrees

    goal_type: str = "RELATIVE_BETWEEN_30_330"  # or "random"

    rotator_baudrate: int = 1000000
    rotator_servo_id: int = 5


class FourFingerConfig(GripperEnvironmentConfig):
    """
    Configuration for the Four Finger environment.
    Inherits from GripperEnvironmentConfig.
    """

    domain: str = "four_finger"


class FourFingerTranslationConfig(FourFingerConfig):
    """
    Configuration for the Four Finger Translation environment.
    Inherits from FourFingerConfig.
    """

    task: str = "translation"

    goal_min: list[float] = [40, 40]  # mm
    goal_max: list[float] = [120, 120]  # mm

    noise_tolerance: float = 10.0  # mm
