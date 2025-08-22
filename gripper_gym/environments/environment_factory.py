from gripper_gym.environments.environment import Environment
from gripper_gym.environments.two_finger.rotation.rotation import TwoFingerRotation
from gripper_gym.environments.two_finger.translation.translation_flat import (
    TwoFingerTranslationFlat,
)
from gripper_gym.environments.two_finger.translation.translation_suspended import (
    TwoFingerTranslationSuspended,
)

from gripper_gym.environments.four_finger.translation.translation_flat import (
    FourFingerTranslationFlat,
)

from gripper_gym.environments.four_finger.rotation.rotation_flat import (
    FourFingerRotationFlat,
)
from gripper_gym.environments.four_finger.rotation.rotation_suspended import (
    FourFingerRotationSuspended,
)


class EnvironmentFactory:
    def __init__(self):
        pass

    def create_environment(
        self, domain: str, task: str, gripper_id: int
    ) -> Environment:
        """
        Create an environment based on the domain and task.

        Args:
        domain: The domain of the environment.
        task: The task of the environment.

        Returns:
        Environment: The environment object.
        """

        environment: Environment | None = None
        if domain == "two_finger":
            if task == "translation":
                environment = TwoFingerTranslationFlat(gripper_id)
            elif task == "suspended_translation":
                environment = TwoFingerTranslationSuspended(gripper_id)
            elif task == "rotation":
                environment = TwoFingerRotation(gripper_id)
        if domain == "four_finger":
            if task == "translation":
                environment = FourFingerTranslationFlat(gripper_id)
            elif task == "rotation":
                environment = FourFingerRotationFlat(gripper_id)
            elif task == "suspended_rotation":
                environment = FourFingerRotationSuspended(gripper_id)

        if environment is None:
            raise ValueError(f"Invalid domain or task: {domain}, {task}")

        return environment
