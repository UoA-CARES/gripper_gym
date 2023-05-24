import logging
logging.basicConfig(level=logging.INFO)

import os
import pydantic

from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

import torch
import random
import numpy as np

from environments.RotationEnvironment import RotationEnvironment
from environments.TranslationEnvironment import TranslationEnvironment
from configurations import LearningConfig, EnvironmentConfig, GripperConfig,ObjectConfig

from environments.Environment import EnvironmentError
from Gripper import GripperError
from tools.utils import create_directories, plot_curve
import tools.error_handlers as erh

from cares_reinforcement_learning.algorithm import TD3
from networks import Actor
from networks import Critic
from cares_reinforcement_learning.util import MemoryBuffer
from cares_lib.slack_bot.SlackBot import SlackBot

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logging.info("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    logging.info("Working with CPU")
 
with open('slack_token.txt') as file: 
    slack_token = file.read()
slack_bot = SlackBot(slack_token=slack_token)


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
        actor  = Actor(observation_size, action_num, learning_config.actor_lr)
        critic = Critic(observation_size, action_num, learning_config.critic_lr)

        logging.info("Setting up Memory")
        self.memory = MemoryBuffer(learning_config.buffer_capacity)

        logging.info("Setting RL Algorithm")
        self.agent = TD3(
            actor_network=actor,
            critic_network=critic,
            gamma=learning_config.gamma,
            tau=learning_config.tau,
            action_num=action_num,
            device=DEVICE,
        )

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

    def evaluation(self):
        logging.info("Starting Training Loop")
        self.agent.load_models(self.file_name, self.file_path)

        episode_timesteps = 0
        episode_reward    = 0
        episode_num       = 0

        state = self.environment_reset() 

        max_steps_evaluation        = 100
        episode_horizont_evaluation = 20

        for total_step_counter in range(max_steps_evaluation):
            episode_timesteps += 1

            action = self.agent.select_action_from_policy(state, evaluation=True)  # algorithm range [-1, 1]
            action_env = self.environment.denormalize(action)  # gripper range

            next_state, reward, done, truncated = self.envrionment.step(action_env)
            if not truncated:
                logging.info(f"Reward of this step:{reward}\n")
                state = next_state
                episode_reward += reward

            if done or truncated or episode_timesteps >= episode_horizont_evaluation:
                logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

                state =  self.environment_reset() 

                episode_reward    = 0
                episode_timesteps = 0
                episode_num += 1

    def train(self):
        logging.info("Starting Training Loop")

        episode_timesteps = 0
        episode_reward    = 0
        episode_num       = 0

        historical_reward = {"step": [], "episode_reward": []}
        historical_distance = {"step": [], "episode_distance": []}
        historical_success_rate = {"step": [], "episode_success_rate": []}
        
        best_episode_reward = -np.inf
        success_count = 0

        state = self.environment_reset() 

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
                noise_scale *= self.noise_decay
                noise_scale = max(self.min_noise, self.noise_scale)
                # logging.info(f"Noise Scale:{noise_scale}")

                message = f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n"
                logging.info(message)

                action = self.agent.select_action_from_policy(state, noise_scale=noise_scale)  # algorithm range [-1, 1]
                action_env = self.environment.denormalize(action)  # gripper range
            
            next_state, reward, done, truncated = self.environment_step(action_env)

            if not truncated:
                logging.info(f"Reward of this step:{reward}\n")

                self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

                state = next_state

                episode_reward += reward

                if self.environment.action_type == "position": # if position based, train every step
                    if total_step_counter >= self.max_steps_exploration:
                        for _ in range(self.G):
                            experiences = self.memory.sample(self.batch_size)
                            self.agent.train_policy(experiences)


                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    self.agent.save_models(f"best_{self.file_name}", self.file_path)

            success_count = (success_count+1) if done else success_count
                        
            if done or truncated or episode_timesteps >= self.episode_horizont:
                message = f"#{self.environment.gripper.gripper_id} - Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}"
                logging.info(message)
                slack_bot.post_message("#bot_terminal", message)

                historical_reward["step"].append(total_step_counter)
                historical_reward["episode_reward"].append(episode_reward)

                episode_distance = self.environment.ep_final_distance()
                historical_distance["step"].append(total_step_counter)
                historical_distance["episode_distance"].append(episode_distance)

                historical_success_rate["step"].append(total_step_counter)
                historical_success_rate["episode_success_rate"].append(success_count/episode_num)

                if self.environment.action_type == "velocity": # if velocity based, train every episode
                    if total_step_counter >= self.max_steps_exploration:
                        for _ in range(self.G):
                            experiences = self.memory.sample(self.batch_size)
                            self.agent.train_policy(experiences)

                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    self.agent.save_models(self.file_name, self.file_path)

                state = self.environment_reset() 

                episode_reward    = 0
                episode_timesteps = 0
                episode_num      += 1

                if episode_num % self.plot_freq == 0:
                    plot_curve(historical_reward, self.file_path, "reward")
                    plot_curve(historical_distance, self.file_path, "distance")
                    plot_curve(historical_success_rate, self.file_path, "success_rate")
                    logging.info(f"Success Rate: {success_count/episode_num} with total {success_count} successes out of {episode_num} episodes")

        plot_curve(historical_reward, self.file_path, "reward")
        plot_curve(historical_distance, self.file_path, "distance")
        self.agent.save_models(self.file_name, self.file_path)
        self.environment.gripper.close()

def store_configs(file_path, env_config, gripper_config, learning_config, object_config):
    with open(f"{file_path}/configs.txt", "w") as f:
        f.write(f"Environment Config:\n{env_config.json()}\n")
        f.write(f"Gripper Config:\n{gripper_config.json()}\n")
        f.write(f"Learning Config:\n{learning_config.json()}\n")
        f.write(f"Learning Config:\n{object_config.json()}\n")
        with open(Path(env_config.camera_matrix)) as cm:
            f.write(f"\nCamera Matrix:\n{cm.read()}\n")
        with open(Path(env_config.camera_distortion)) as cd:
            f.write(f"Camera Distortion:\n{cd.read()}\n")
        
def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--learning_config", type=str)
    parser.add_argument("--env_config",      type=str)
    parser.add_argument("--gripper_config",  type=str)
    parser.add_argument("--object_config",   type=str)

    home_path = os.path.expanduser('~')
    parser.add_argument("--local_results_path",  type=str, default=f"{home_path}/gripper_training")
    return parser.parse_args()

def main():

    args = parse_args()
    env_config      = pydantic.parse_file_as(path=args.env_config,      type_=EnvironmentConfig)
    gripper_config  = pydantic.parse_file_as(path=args.gripper_config,  type_=GripperConfig)
    learning_config = pydantic.parse_file_as(path=args.learning_config, type_=LearningConfig)
    object_config = pydantic.parse_file_as(path=args.object_config, type_=ObjectConfig)
    local_results_path = args.local_results_path

    logging.info("Setting up Seeds")
    torch.manual_seed(learning_config.seed)
    np.random.seed(learning_config.seed)
    random.seed(learning_config.seed)

    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_path  = f"{date_time_str}_"
    file_path += f"RobotId{gripper_config.gripper_id}_EnvType{env_config.env_type}_ObsType{object_config.object_type}_Seed{learning_config.seed}_{learning_config.algorithm}"

    file_path = create_directories(local_results_path, file_path)
    store_configs(file_path, env_config, gripper_config, learning_config, object_config)

    gripper_trainer = GripperTrainer(env_config, gripper_config, learning_config, object_config, file_path)
    gripper_trainer.train()

    #gripper_trainner.evaluation()

if __name__ == '__main__':
    main()