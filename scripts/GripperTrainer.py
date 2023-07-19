import logging
logging.basicConfig(level=logging.INFO)

import time
import torch
import os
import numpy as np
import tools.utils as utils
import tools.error_handlers as erh

from datetime import datetime
from collections import deque

from environments.RotationEnvironment import RotationEnvironment
from environments.TranslationEnvironment import TranslationEnvironment

from environments.Environment import EnvironmentError
from Gripper import GripperError

from networks import NetworkFactory
from cares_reinforcement_learning.memory import MemoryBuffer

from cares_lib.slack_bot.SlackBot import SlackBot
from enum import Enum


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
        
        if env_config.env_type == 0:
            self.environment = RotationEnvironment(env_config, gripper_config, object_config)
        elif env_config.env_type == 1:
            self.environment = TranslationEnvironment(env_config, gripper_config, object_config)

        logging.info("Resetting Environment")
        state = self.environment.reset()#will just crash right away if there is an issue but that is fine
        
        logging.info(f"State: {state}")
        slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: Reset Terminal. \nState: {state}")

        observation_size = len(state)# This wont work for multi-dimension arrays
        action_num       = gripper_config.num_motors
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
        try:
            return self.environment.reset()
        except (EnvironmentError , GripperError) as error:
            error_message = f"Failed to reset with message: {error}"
            logging.error(error_message)
            if erh.handle_gripper_error_home(self.environment, error_message, slack_bot, self.file_path):
                return self.environment_reset()  # might keep looping if it keep having issues
            else:
                self.environment.gripper.close()
                self.agent.save_models(self.file_name, self.file_path)
                exit(1)

    def environment_step(self, action_env):
        try:
            return self.environment.step(action_env)
        except (EnvironmentError , GripperError) as error:
            error_message = f"Failed to step environment with message: {error}"
            logging.error(error_message)
            if erh.handle_gripper_error_home(self.environment, error_message, slack_bot, self.file_path):
                return [], 0, False, True
            else:
                self.environment.gripper.close()
                self.agent.save_models(self.file_name, self.file_path)
                exit(1)

    def evaluate(self, model_path):
        logging.info("Starting Evaluation Loop")

        model_name = f"best_{os.path.basename(os.path.normpath(model_path))}"
        self.agent.load_models(model_path, model_name)

        episode_timesteps = 0
        episode_reward    = 0
        episode_num       = 0

        success_window_size = 100 #episodes
        steps_per_episode_window_size = 5 #episodes
        rolling_success_rate = deque(maxlen=success_window_size)
        rolling_reward_rate  = deque(maxlen=success_window_size)
        rolling_steps_per_episode = deque(maxlen=steps_per_episode_window_size)
        plots = ["reward", "distance", "rolling_success_average", "rolling_reward_average", "rolling_steps_per_episode_average"]
        previous_T_step = 0

        state = self.environment_reset() 

        max_steps_evaluation        = 1000
        episode_horizont_evaluation = 50

        for total_step_counter in range(max_steps_evaluation):
            episode_timesteps += 1
            message = f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n"
            logging.info(message)

            self.noise_scale *= self.noise_decay
            self.noise_scale = max(self.min_noise, self.noise_scale)

            if (self.algorithm == ALGORITHMS.TD3.value):
                action = self.agent.select_action_from_policy(state, noise_scale=self.noise_scale)  # returns a 1D array with range [-1, 1], only TD3 has noise scale
            else:
                action = self.agent.select_action_from_policy(state)

            action_env = self.environment.denormalize(action)  # gripper range

            next_state, reward, done, truncated = self.environment.step(action_env)

            if self.environment.action_type == "velocity":
                self.environment.step_gripper()
            
            if not truncated:
                logging.info(f"Reward of this step:{reward}\n")
                state = next_state
                episode_reward += reward

            if done or truncated or episode_timesteps >= episode_horizont_evaluation:
                logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

                state =  self.environment_reset() 

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

                # --- Storing steps per episode data ---
                steps_per_episode = total_step_counter - previous_T_step
                previous_T_step = total_step_counter
                utils.store_data(steps_per_episode, self.file_path, "steps_per_episode")
                rolling_steps_per_episode.append(steps_per_episode)

                rolling_steps_per_episode_average = sum(rolling_steps_per_episode)/len(rolling_steps_per_episode)
                utils.store_data(rolling_steps_per_episode_average, self.file_path, "rolling_steps_per_episode_average")

                episode_reward    = 0
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
        logging.info("Starting Training Loop")

        episode_timesteps = 0
        episode_reward    = 0
        episode_num       = 0
        start_time = datetime.now()
        
        success_window_size = 100 #episodes
        steps_per_episode_window_size = 5 #episodes
        rolling_success_rate = deque(maxlen=success_window_size)
        rolling_reward_rate  = deque(maxlen=success_window_size)
        rolling_steps_per_episode = deque(maxlen=steps_per_episode_window_size)
        plots = ["reward", "distance", "rolling_success_average", "rolling_reward_average", "rolling_steps_per_episode_average"]
        time_plots = ["reward_average_vs_time"]
        
        best_episode_reward = -np.inf
        previous_T_step = 0

        state = self.environment_reset()

        prev_time = time.time()

        for total_step_counter in range(int(self.max_steps_training)):
            episode_timesteps += 1

            if total_step_counter < self.max_steps_exploration:
                message = f"Running Exploration Steps {total_step_counter}/{self.max_steps_exploration}"
                logging.info(message)
                if total_step_counter%50 == 0:
                    slack_bot.post_message("#bot_terminal", f"#{self.environment.gripper.gripper_id}: {message}")
                
                action_env = self.environment.sample_action()
                action = self.environment.normalize(action_env) # algorithm range [-1, 1]
            else:
                self.noise_scale *= self.noise_decay
                self.noise_scale = max(self.min_noise, self.noise_scale)
                # logging.info(f"Noise Scale:{self.noise_scale}")

                message = f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n"
                logging.info(message)

                if (self.algorithm == ALGORITHMS.TD3.value):
                    action = self.agent.select_action_from_policy(state, noise_scale=self.noise_scale)  # returns a 1D array with range [-1, 1], only TD3 has noise scale
                else:
                    action = self.agent.select_action_from_policy(state)

                action_env = self.environment.denormalize(action)  # gripper range

            env_start = time.time()
            next_state, reward, done, truncated = self.environment_step(action_env)
            env_end = time.time()
            logging.info(f"time to execute environment_step: {env_end-env_start}")
            
            if self.environment.action_type == "velocity":
                # time between this being called each loop...
                self.environment.step_gripper()
                logging.info(f"Time since step_gripper was last called: {time.time() - prev_time}")
                prev_time = time.time()

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
                logging.info(f"Time to run training loop {end-start} \n")

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
                utils.store_data(round(episode_time.total_seconds()), self.file_path, "time") # Stores time in seconds since beginning training

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

                episode_reward    = 0
                episode_timesteps = 0
                episode_num      += 1

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