from environments.two_finger.translation import (
    TwoFingerTranslationFlat,
    TwoFingerTranslationSuspended,
)
from environments.two_finger.rotation import TwoFingerRotationTask
from environments.four_finger.rotation import FourFingerRotationFlat
from environments.four_finger.rotation import FourFingerRotationSuspended


class EnvironmentFactory:
    def __init__(self):
        pass

    def create_environment(self, env_config, gripper_config):
        """
        Create an environment based on the domain and task.

        Args:
        domain: The domain of the environment.
        task: The task of the environment.

        Returns:
        Environment: The environment object.
        """
        domain = env_config.domain
        task = env_config.task
        
        environment = None
        if domain == "two_finger":
            if task == "translation":
                environment = TwoFingerTranslationFlat(env_config, gripper_config)
            elif task == "suspended_translation":
                environment = TwoFingerTranslationSuspended(env_config, gripper_config)
            elif task == "rotation":
                environment = TwoFingerRotationTask(env_config, gripper_config)
        if domain == "four_finger":
            if task == "translation":
                environment = None
            elif task == "rotation":
                environment = FourFingerRotationFlat(env_config, gripper_config)
            elif task == "suspended_rotation":
                environment = FourFingerRotationSuspended(env_config, gripper_config)


        if environment is None:
            raise ValueError(f"Invalid domain or task: {domain}, {task}")

        return environment
