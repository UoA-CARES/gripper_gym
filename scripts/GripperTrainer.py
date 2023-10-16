from enum import Enum
from cares_lib.slack_bot.SlackBot import SlackBot
from cares_reinforcement_learning.memory import MemoryBuffer
from networks import NetworkFactory
from cares_lib.dynamixel.Gripper import GripperError
from environments.Environment import EnvironmentError
from environments.TranslationEnvironment import TranslationEnvironment
from environments.RotationEnvironment import RotationEnvironment
from collections import deque
from datetime import datetime
import tools.error_handlers as erh
import tools.utils as utils
import numpy as np
import os
import torch
import time
import logging
logging.basicConfig(level=logging.INFO)


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logging.info("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    logging.info("Working with CPU")

with open('slack_token.txt') as file:
    slack_token = file.read()
slack_bot = SlackBot(slack_token=slack_token)


class ALGORITHMS(Enum):
    TD3 = "TD3"
    SAC = "SAC"
    DDPG = "DDPG"


class GripperTrainer():
    def __init__(self, env_config, gripper_config, learning_config, object_config, file_path) -> None:
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
        self.seed = learning_config.seed
        self.batch_size = learning_config.batch_size
        self.buffer_capacity = learning_config.buffer_capacity
        self.episode_horizont = learning_config.episode_horizont

        self.G = learning_config.G
        self.plot_freq = learning_config.plot_freq

        self.max_steps_exploration = learning_config.max_steps_exploration
        self.max_steps_training = learning_config.max_steps_training

        self.actor_lr = learning_config.actor_lr
        self.critic_lr = learning_config.critic_lr
        self.gamma = learning_config.gamma
        self.tau = learning_config.tau

        self.min_noise = learning_config.min_noise
        self.noise_decay = learning_config.noise_decay
        self.noise_scale = learning_config.noise_scale
        self.algorithm = learning_config.algorithm

        self.action_type = gripper_config.action_type

        self.eval_freq = 10  # evaluate every 10 episodes

        if env_config.env_type == 0:
            self.environment = RotationEnvironment(env_config, gripper_config, object_config)
        elif env_config.env_type == 1:
            self.environment = TranslationEnvironment(env_config, gripper_config, object_config)

        logging.info("Resetting Environment")
        # will just crash right away if there is an issue but that is fine
        state = self.environment.reset()

        logging.info(f"State: {state}")
        slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: Reset Terminal. \nState: {state}")

        # This wont work for multi-dimension arrays
        observation_size = len(state)
        action_num = gripper_config.num_motors
        message = f"Observation Space: {observation_size} Action Space: {action_num}"
        logging.info(message)
        slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: {message}")

        logging.info("Setting up Network")
        network_factory = NetworkFactory()

        logging.info("Setting up Memory")
        self.memory = MemoryBuffer(learning_config.buffer_capacity)

        logging.info("Setting RL Algorithm")
        logging.info(f"Chosen algorithm: {self.algorithm}")
        self.agent = network_factory.create_network(self.algorithm, observation_size, action_num, learning_config, DEVICE)

        self.file_path = file_path
        self.file_name = self.file_path.split("/")[-1]

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
            if erh.handle_gripper_error_home(self.environment, error_message, slack_bot, self.file_path):
                return self.environment_reset()  # might keep looping if it keep having issues
            else:
                self.environment.gripper.close()
                self.agent.save_models(self.file_name, self.file_path)
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
            if erh.handle_gripper_error_home(self.environment, error_message, slack_bot, self.file_path):
                state = self.environment.get_state()
                # Truncated should be false to prevent skipping the entire episode
                return state, 0, False, False
            else:
                self.environment.gripper.close()
                self.agent.save_models(self.file_name, self.file_path)
                exit(1)

    def evaluation_loop(self, total_counter, file_name, historical_reward_evaluation):
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
        historical_episode_reward_evaluation = []

        for total_step_counter in range(max_steps_evaluation):
            episode_timesteps += 1
            message = f"EVALUATION | Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n"
            logging.info(message)

            self.noise_scale *= self.noise_decay
            self.noise_scale = max(self.min_noise, self.noise_scale)

            if (self.algorithm == ALGORITHMS.TD3.value):
                # returns a 1D array with range [-1, 1], only TD3 has noise scale
                action = self.agent.select_action_from_policy(
                    state, noise_scale=self.noise_scale)
            else:
                action = self.agent.select_action_from_policy(state)

            action_env = self.environment.denormalize(action)  # gripper range

            next_state, reward, done, truncated = self.environment_step(action_env)

            if self.environment.action_type == "velocity":
                try:
                    self.environment.step_gripper()
                except (EnvironmentError, GripperError):
                    continue

            if not truncated:
                logging.info(f"EVALUATION | Reward of this step:{reward}\n")
                state = next_state
                episode_reward += reward

            if done or truncated or episode_timesteps >= max_steps_evaluation:
                logging.info(f"EVALUATION | Eval Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f}")
                historical_episode_reward_evaluation.append(episode_reward)

                state = self.environment_reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        # at end of evaluation, save the data
        mean_reward_evaluation = np.round(np.mean(historical_episode_reward_evaluation), 2)
        historical_reward_evaluation["avg_episode_reward"].append(mean_reward_evaluation)
        historical_reward_evaluation["step"].append(total_counter)
        utils.save_evaluation_values(historical_reward_evaluation, file_name, self.file_path)

    def evaluate_at_end(self, model_path):
        """
        Evaluates the performance of an agent model at the end of training.

        Args:
        model_path (str): Path to the trained model to be evaluated.

        The method is intended for post-training evaluation, providing insights on the agent's performance.
        """
        logging.info("Starting Evaluation Loop")

        model_name = f"best_{os.path.basename(os.path.normpath(model_path))}"
        self.agent.load_models(model_path, model_name)

        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0

        success_window_size = 100  # episodes
        steps_per_episode_window_size = 5  # episodes
        rolling_success_rate = deque(maxlen=success_window_size)
        rolling_reward_rate = deque(maxlen=success_window_size)
        rolling_steps_per_episode = deque(maxlen=steps_per_episode_window_size)
        plots = ["reward", "distance", "rolling_success_average", "rolling_reward_average", "rolling_steps_per_episode_average"]
        previous_T_step = 0

        state = self.environment_reset()

        max_steps_evaluation = 1000
        episode_horizont_evaluation = 50

        for total_step_counter in range(max_steps_evaluation):
            episode_timesteps += 1
            message = f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter}"
            logging.info(message)

            self.noise_scale *= self.noise_decay
            self.noise_scale = max(self.min_noise, self.noise_scale)

            if (self.algorithm == ALGORITHMS.TD3.value):
                # returns a 1D array with range [-1, 1], only TD3 has noise scale
                action = self.agent.select_action_from_policy(state, noise_scale=self.noise_scale)
            else:
                action = self.agent.select_action_from_policy(state)

            action_env = self.environment.denormalize(action)  # gripper range

            next_state, reward, done, truncated = self.environment_step(action_env)

            if not truncated:
                logging.info(f"Reward of this step:{reward}\n")
                state = next_state
                episode_reward += reward

            if done or truncated or episode_timesteps >= episode_horizont_evaluation:
                logging.info(f"EVALUATION | Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

                state = self.environment_reset()

                # --- Storing success data ---
                if done:
                    rolling_success_rate.append(1)
                    utils.store_data("1", self.file_path, "success_list")
                else:
                    rolling_success_rate.append(0)
                    utils.store_data("0", self.file_path, "success_list")
                rolling_success_average = sum(rolling_success_rate)/len(rolling_success_rate)
                utils.store_data(rolling_success_average, self.file_path, "rolling_success_average")

                # --- Storing reward data ---
                utils.store_data(episode_reward, self.file_path, "reward")
                rolling_reward_rate.append(episode_reward)

                rolling_reward_average = sum(rolling_reward_rate)/len(rolling_reward_rate)
                utils.store_data(rolling_reward_average,self.file_path, "rolling_reward_average")

                # --- Storing steps per episode data ---
                steps_per_episode = total_step_counter - previous_T_step
                previous_T_step = total_step_counter
                utils.store_data(steps_per_episode,self.file_path, "steps_per_episode")
                rolling_steps_per_episode.append(steps_per_episode)

                rolling_steps_per_episode_average = sum(rolling_steps_per_episode)/len(rolling_steps_per_episode)
                utils.store_data(rolling_steps_per_episode_average, self.file_path, "rolling_steps_per_episode_average")

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if episode_num % self.plot_freq == 0:
                    average_success_message = f"Average Success Rate: {rolling_success_average} over last {success_window_size} episodes\n"
                    average_reward_message = f"Average Reward: {rolling_reward_average} over last {success_window_size} episodes\n"
                    average_steps_per_episode_message = f"Average Steps Per Episode: {rolling_steps_per_episode_average} over last {steps_per_episode_window_size} episodes\n"

                    logging.info(f"\n{average_success_message}{average_reward_message}{average_steps_per_episode_message}")
                    slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: {average_success_message}{average_reward_message}{average_steps_per_episode_message}")

        self.environment.gripper.close()

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
        start_time = datetime.now()

        success_window_size = 100  # episodes
        steps_per_episode_window_size = 5  # episodes

        env_end = time.time()
        time_period = 0.15
        
        success_window_size = 100 #episodes
        steps_per_episode_window_size = 5 #episodes

        rolling_success_rate = deque(maxlen=success_window_size)
        rolling_reward_rate = deque(maxlen=success_window_size)
        rolling_steps_per_episode = deque(maxlen=steps_per_episode_window_size)
        plots = ["reward", "distance", "rolling_success_average","rolling_reward_average", "rolling_steps_per_episode_average"]
        time_plots = ["reward_average_vs_time"]

        historical_reward_evaluation = {"step": [], "avg_episode_reward": []}

        # To store zero at the beginning
        historical_reward_evaluation["step"].append(0)
        historical_reward_evaluation["avg_episode_reward"].append(0)

        best_episode_reward = -np.inf
        previous_T_step = 0

        state = self.environment_reset()

        for total_step_counter in range(int(self.max_steps_training)):
            episode_timesteps += 1

            if total_step_counter < self.max_steps_exploration:
                message = f"Running Exploration Steps {total_step_counter}/{self.max_steps_exploration}"
                logging.info(message)
                if total_step_counter % 50 == 0:
                    slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: {message}")

                action_env = self.environment.sample_action()
                action = self.environment.normalize(action_env)  # algorithm range [-1, 1]
            else:
                self.noise_scale *= self.noise_decay
                self.noise_scale = max(self.min_noise, self.noise_scale)
                logging.debug(f"Noise Scale:{self.noise_scale}")

                message = f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter}"
                logging.info(message)

                if (self.algorithm == ALGORITHMS.TD3.value):
                    # returns a 1D array with range [-1, 1], only TD3 has noise scale
                    action = self.agent.select_action_from_policy(state, noise_scale=self.noise_scale)
                else:
                    action = self.agent.select_action_from_policy(state)

                action_env = self.environment.denormalize(action)  # gripper range

            env_start = time.time()
            logging.debug(f"time to execute training loop (excluding environment_step): {env_start-env_end} before delay")
            delay = time_period-(env_start-env_end)
            if delay > 0:
               time.sleep(delay)
            
            next_state, reward, done, truncated = self.environment_step(action_env)
            
            env_end = time.time()

            if not truncated:
                logging.info(f"Reward of this step:{reward}\n")

                self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

                state = next_state

                episode_reward += reward

                # Regardless if velocity or position based, train every step
                start = time.time()
                if total_step_counter >= self.max_steps_exploration:
                    for _ in range(self.G):
                        experiences = self.memory.sample(self.batch_size)
                        info = self.agent.train_policy((
                            experiences['state'],
                            experiences['action'],
                            experiences['reward'],
                            experiences['next_state'],
                            experiences['done']
                        ))
                end = time.time()
                logging.debug(f"Time to run training loop {end-start} \n")

                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    self.agent.save_models(f"best_{self.file_name}", self.file_path)

            if done or truncated or episode_timesteps >= self.episode_horizont:
                message = f"#{self.environment.gripper.gripper_id} - Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}"
                logging.info(message)
                slack_bot.post_message("#bot_terminal", message)

                # --- Storing success data ---
                if done:
                    rolling_success_rate.append(1)
                    utils.store_data("1", self.file_path, "success_list")
                else:
                    rolling_success_rate.append(0)
                    utils.store_data("0", self.file_path, "success_list")
                rolling_success_average = sum(rolling_success_rate)/len(rolling_success_rate)
                utils.store_data(rolling_success_average, self.file_path, "rolling_success_average")

                # --- Storing reward data ---
                utils.store_data(episode_reward, self.file_path, "reward")
                rolling_reward_rate.append(episode_reward)

                rolling_reward_average = sum(rolling_reward_rate)/len(rolling_reward_rate)
                utils.store_data(rolling_reward_average, self.file_path, "rolling_reward_average")

                # --- Storing time data ---
                episode_time = datetime.now() - start_time
                # Stores time in seconds since beginning training
                utils.store_data(round(episode_time.total_seconds()), self.file_path, "time")

                # --- Storing distance data ---
                episode_distance = self.environment.ep_final_distance()
                utils.store_data(episode_distance, self.file_path, "distance")

                # --- Storing steps per episode data ---
                steps_per_episode = total_step_counter - previous_T_step
                previous_T_step = total_step_counter
                utils.store_data(steps_per_episode, self.file_path, "steps_per_episode")
                rolling_steps_per_episode.append(steps_per_episode)

                rolling_steps_per_episode_average = sum(rolling_steps_per_episode)/len(rolling_steps_per_episode)
                utils.store_data(rolling_steps_per_episode_average, self.file_path, "rolling_steps_per_episode_average")

                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    self.agent.save_models(self.file_name, self.file_path)

                state = self.environment_reset()

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if episode_num % self.eval_freq == 0:
                    logging.info("*************--Evaluation Loop--*************")
                    self.evaluation_loop(total_step_counter, self.file_name, historical_reward_evaluation)
                    # reset env at end of evaluation before continuing
                    state = self.environment_reset()
                    logging.info("--------------------------------------------")

                if episode_num % (self.plot_freq*10) == 0:
                    utils.plot_data(self.file_path, "reward")
                    utils.plot_data(self.file_path, "distance")

                    utils.slack_post_plot(self.environment, slack_bot, self.file_path, plots)
                    utils.slack_post_plot(self.environment, slack_bot, self.file_path, time_plots)

                if episode_num % self.plot_freq == 0:
                    utils.plot_data(self.file_path, "rolling_success_average")
                    utils.plot_data(self.file_path, "rolling_reward_average")
                    utils.plot_data(self.file_path, "rolling_steps_per_episode_average")
                    utils.plot_data_time(self.file_path, "time", "rolling_reward_average", "time")

                    average_success_message = f"Average Success Rate: {rolling_success_average} over last {success_window_size} episodes\n"
                    average_reward_message = f"Average Reward: {rolling_reward_average} over last {success_window_size} episodes\n"
                    average_steps_per_episode_message = f"Average Steps Per Episode: {rolling_steps_per_episode_average} over last {steps_per_episode_window_size} episodes\n"

                    logging.info(f"\n{average_success_message}{average_reward_message}{average_steps_per_episode_message}")
                    slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: {average_success_message}{average_reward_message}{average_steps_per_episode_message}")

        utils.plot_data(self.file_path, plots)
        utils.plot_data_time(self.file_path, "time", "rolling_reward_average", "time")
        self.agent.save_models(self.file_name, self.file_path)
        self.environment.gripper.close()
