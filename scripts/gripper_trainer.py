import logging
import time
from datetime import datetime

import tools.error_handlers as erh
from cares_lib.dynamixel.Gripper import GripperError
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from configurations import GripperEnvironmentConfig, ObjectConfig
from environments.environment import EnvironmnetFactory
from networks.NetworkFactory import NetworkFactory

logging.basicConfig(level=logging.INFO)


class GripperTrainer:
    def __init__(
        self,
        env_config: GripperEnvironmentConfig,
        training_config: TrainingConfig,
        alg_config: AlgorithmConfig,
        gripper_config: GripperConfig,
        object_config: ObjectConfig,
    ) -> None:
        """
        Initializes the GripperTrainer class for training gripper actions in various environments.

        Args:
        env_config (Object): Configuration parameters for the environment.
        gripper_config (Object): Configuration parameters for the gripper.
        learning_config (Object): Configuration parameters for the learning/training process.
        object_config (Object): Configuration parameters for the objects in the environment.
        file_path (str): Path to save or retrieve model related files.

        Many of the instances variables are initialised from the provided configurations and are used throughout the training process.
        """

        self.seed = training_config.seeds[0]  # TODO: reconcile the multiple seeds
        self.batch_size = training_config.batch_size

        self.G = training_config.G
        self.plot_freq = training_config.plot_frequency

        self.max_steps_exploration = training_config.max_steps_exploration
        self.max_steps_training = training_config.max_steps_training
        self.number_steps_per_evaluation = training_config.number_steps_per_evaluation
        self.number_eval_episodes = training_config.number_eval_episodes

        self.step_time_period = env_config.step_length

        self.action_type = gripper_config.action_type

        env_factory = EnvironmnetFactory()
        self.environment = env_factory.create_environment(
            env_config, gripper_config, object_config
        )

        logging.info("Resetting Environment")
        # will just crash right away if there is an issue but that is fine
        state = self.environment.reset()

        logging.info(f"State: {state}")

        # This wont work for multi-dimension arrays
        observation_size = len(state)
        action_num = gripper_config.num_motors
        logging.info(
            f"Observation Space: {observation_size} Action Space: {action_num}"
        )

        network_factory = NetworkFactory()
        self.agent = network_factory.create_network(
            observation_size, action_num, alg_config
        )

        self.memory = MemoryBuffer(training_config.buffer_size)

        # TODO: reconcile deep file_path dependency
        self.file_path = f'{datetime.now().strftime("%Y_%m_%d_%H:%M:%S")}-gripper-{gripper_config.gripper_id}-{env_config.task}-{alg_config.algorithm}'
        self.record = Record(
            glob_log_dir="../gripper-training",
            log_dir=self.file_path,
            algorithm=self.algorithm,
            task=env_config.task,
            plot_frequency=self.plot_freq,
            network=self.agent,
        )

        self.record.save_config(env_config, "env_config")
        self.record.save_config(alg_config, "alg_config")
        self.record.save_config(training_config, "training_config")
        self.record.save_config(gripper_config, "gripper_config")
        self.record.save_config(object_config, "object_config")

    def environment_reset(self):
        """
        Attempts to reset the environment and handle any encountered errors.

        Returns:
        The result of `self.environment.reset()` if successful.

        Raises:
        EnvironmentError: If there's an error related to the environment during reset.
        GripperError: If there's an error related to the gripper during reset.
        """
        try:
            return self.environment.reset()
        except (EnvironmentError, GripperError) as error:
            error_message = f"Failed to reset with message: {error}"
            logging.error(error_message)
            if erh.handle_gripper_error_home(
                self.environment, error_message, self.file_path
            ):
                return (
                    self.environment_reset()
                )  # might keep looping if it keep having issues
            else:
                self.environment.gripper.close()
                self.agent.save_models("error_models", self.file_path)
                exit(1)

    def environment_step(self, action_env):
        """
        Executes a step in the environment using the given action and handles potential errors.

        Args:
        action_env: The action to be executed in the environment.

        Returns:
        The result of `self.environment.step(action_env)` if successful, which is in the form of (state, reward, done, info).
        If an error is encountered but handled, returns (state, 0, False, False).

        Raises:
        EnvironmentError: If there's an error related to the environment during the step.
        GripperError: If there's an error related to the gripper during the step.
        """
        try:
            return self.environment.step(action_env)
        except (EnvironmentError, GripperError) as error:
            error_message = f"Failed to step environment with message: {error}"
            logging.error(error_message)
            if erh.handle_gripper_error_home(
                self.environment, error_message, self.file_path
            ):
                state = self.environment.get_object_pose()
                # Truncated should be false to prevent skipping the entire episode
                return state, 0, False, False
            else:
                self.environment.gripper.close()
                self.agent.save_models("error_models", self.file_path)
                exit(1)

    def evaluation_loop(self, total_counter):
        """
        Executes an evaluation loop to assess the agent's performance.

        Args:
        total_counter: The total step count leading up to the current evaluation loop.
        file_name: Name of the file where evaluation results will be saved.
        historical_reward_evaluation: Historical rewards list that holds average rewards from previous evaluations.

        The method aims to evaluate the agent's performance by running the environment for a set number of steps and recording the average reward.
        """
        max_steps_evaluation = 50
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0

        state = self.environment_reset()
        env_end = time.time()

        for _ in range(max_steps_evaluation):
            episode_timesteps += 1

            self.noise_scale *= self.noise_decay
            self.noise_scale = max(self.min_noise, self.noise_scale)

            if self.algorithm == "TD3":
                # returns a 1D array with range [-1, 1], only TD3 has noise scale
                action = self.agent.select_action_from_policy(
                    state, noise_scale=self.noise_scale, evaluation=True
                )
            else:
                action = self.agent.select_action_from_policy(state, evaluation=True)

            action_env = self.environment.denormalize(action)  # gripper range

            if self.action_type == "velocity":
                self.dynamic_sleep(env_end)

            next_state, reward, done, truncated = self.environment_step(action_env)

            env_end = time.time()

            if self.environment.action_type == "velocity":
                try:
                    self.environment.step_gripper()
                except (EnvironmentError, GripperError):
                    continue

            state = next_state
            episode_reward += reward

            if done or truncated:

                self.record.log_eval(
                    total_steps=total_counter + 1,
                    episode=episode_num,
                    episode_steps=episode_timesteps,
                    episode_reward=episode_reward,
                    display=True,
                )

                state = self.environment_reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

    def train(self):
        """
        This method is the main training loop that is called to start training the agent a given environment.
        Trains the agent and save the results in a file and periodically evaluate the agent's performance as well as plotting results.
        Logging and messaging to a specified Slack Channel for monitoring the progress of the training.
        """
        logging.info("Starting Training Loop")

        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        evaluation = False

        start_time = datetime.now()

        env_end = time.time()

        state = self.environment_reset()

        for total_step_counter in range(int(self.max_steps_training)):
            episode_timesteps += 1

            if total_step_counter < self.max_steps_exploration:
                message = f"Running Exploration Steps {total_step_counter}/{self.max_steps_exploration}"
                logging.info(message)

                action_env = self.environment.sample_action()

                # algorithm range [-1, 1]
                action = self.environment.normalize(action_env)
            else:
                self.noise_scale *= self.noise_decay
                self.noise_scale = max(self.min_noise, self.noise_scale)
                logging.debug(f"Noise Scale:{self.noise_scale}")

                if self.algorithm == ALGORITHMS.TD3.value:
                    # returns a 1D array with range [-1, 1], only TD3 has noise scale
                    action = self.agent.select_action_from_policy(
                        state, noise_scale=self.noise_scale
                    )
                else:
                    action = self.agent.select_action_from_policy(state)

                # gripper range
                action_env = self.environment.denormalize(action)

            if self.action_type == "velocity":
                self.dynamic_sleep(env_end)

            next_state, reward, done, truncated = self.environment_step(action_env)

            env_end = time.time()

            self.memory.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # Regardless if velocity or position based, train every step
            start = time.time()
            if total_step_counter >= self.max_steps_exploration:
                for _ in range(self.G):
                    experiences = self.memory.sample(self.batch_size)
                    info = self.agent.train_policy(experiences)
            end = time.time()
            logging.debug(f"Time to run training loop {end-start} \n")

            if total_step_counter % self.number_steps_per_evaluation == 0:
                evaluation = True

            if done or truncated:
                self.record.log_train(
                    total_steps=total_step_counter,
                    episode=episode_num,
                    episode_steps=episode_timesteps,
                    episode_reward=episode_reward,
                    display=True,
                )

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if evaluation:
                    logging.info("*************--Evaluation Loop--*************")
                    self.evaluation_loop(total_step_counter)
                    evaluation = False
                    logging.info("--------------------------------------------")

                state = self.environment_reset()

    def dynamic_sleep(self, env_end):
        env_start = time.time()
        logging.debug(
            f"time to execute training loop (excluding environment_step): {env_start-env_end} before delay"
        )

        delay = self.step_time_period - (env_start - env_end)
        if delay > 0:
            time.sleep(delay)
