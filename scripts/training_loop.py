import logging
logging.basicConfig(level=logging.INFO)

import os
import pydantic
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from argparse import ArgumentParser
from pytimedinput import timedInput
from datetime import datetime

import torch
import random
import numpy as np

from environments.RotationEnvironment import RotationEnvironment
from environments.TranslationEnvironment import TranslationEnvironment
from configurations import LearningConfig, EnvironmentConfig, GripperConfig

from environments.Environment import EnvironmentError
from Gripper import GripperError

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


def evaluation(environment, agent, file_name):
    agent.load_models(filename=file_name)

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state = environment_reset(environment) 
    # state = scaling_symlog(state)

    max_steps_evaluation        = 100
    episode_horizont_evaluation = 20

    for total_step_counter in range(max_steps_evaluation):
        episode_timesteps += 1

        action = agent.select_action_from_policy(state, evaluation=True)  # algorithm range [-1, 1]
        action_env = environment.denormalize(action)  # gripper range

        try:
            next_state, reward, done, truncated = environment.step(action_env)

            logging.info(f"Reward of this step:{reward}")
            state = next_state
            episode_reward += reward

        except (EnvironmentError , GripperError) as error:
            error_message = f"Failed to step with message: {error}"
            logging.error(error_message)
            if handle_gripper_error_home(environment, error_message):
                done = True
            else:
                environment.gripper.close() # can do more if we want to save states and all

        if done is True or episode_timesteps >= episode_horizont_evaluation:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            # Reset environment
            state =  environment_reset(environment)

            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1



def train(environment, agent, memory, learning_config, file_name):

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0
    historical_reward = {"step": [], "episode_reward": []}

    min_noise    = 0.01
    noise_decay  = 0.9999
    noise_scale  = 0.10

    state = environment_reset(environment)
    for total_step_counter in range(int(learning_config.max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < learning_config.max_steps_exploration:
            message = f"Running Exploration Steps {total_step_counter}/{learning_config.max_steps_exploration}"
            logging.info(message)

            if total_step_counter%50 == 0:
                slack_bot.post_message("#bot_terminal", message)
            
            if environment.observation_type == 3:
                action_env = environment.sample_action_velocity() # gripper range #or sample_action_velocity
            else:
                action_env = environment.sample_action()
            action     = environment.normalize(action_env) # algorithm range [-1, 1]
        else:
            noise_scale *= noise_decay
            noise_scale = max(min_noise, noise_scale)
            logging.info(f"Noise Scale:{noise_scale}")

            message = f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n"
            logging.info(message)

            action = agent.select_action_from_policy(state, noise_scale=noise_scale)  # algorithm range [-1, 1]
            action_env = environment.denormalize(action)  # gripper range

        try:
            next_state, reward, done, truncated = environment.step(action_env)

            logging.info(f"Reward of this step:{reward}")

            # next_state = scaling_symlog(next_state)

            memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

            state = next_state

            episode_reward += reward

            if total_step_counter >= learning_config.max_steps_exploration:
                if environment.observation_type != 3: # if not velocity based, train every step
                    for _ in range(learning_config.G):
                        experiences = memory.sample(learning_config.batch_size)
                        agent.train_policy(experiences)
                    
        except (EnvironmentError , GripperError) as error:
            error_message = f"Failed to step with message: {error}"
            logging.error(error_message)
            if handle_gripper_error_home(environment, error_message):
                done = True
            else:
                environment.gripper.close() # can do more if we want to save states and all
                break

        if done is True or episode_timesteps >= learning_config.episode_horizont:
            message = f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}"
            logging.info(message)
            slack_bot.post_message("#bot_terminal", message)

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            if total_step_counter >= learning_config.max_steps_exploration:
                if environment.observation_type == 3: # if velocity based, train every episode
                    for _ in range(learning_config.G):
                        experiences = memory.sample(learning_config.batch_size)
                        agent.train_policy(experiences)

            state = environment_reset(environment)
            # state = scaling_symlog(state)

            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

            if episode_num % learning_config.plot_freq == 0:
                plot_reward_curve(historical_reward, file_name)

    plot_reward_curve(historical_reward, file_name)
    agent.save_models(file_name)

# todo move this function to better place
def create_directories():
    if not os.path.exists("./results_plots"):
        os.makedirs("./results_plots")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_plots"):
        os.makedirs("./data_plots")
    if not os.path.exists("./servo_errors"):
        os.makedirs("./servo_errors")

# todo move this function to better place
def plot_reward_curve(data_reward, filename):
    data = pd.DataFrame.from_dict(data_reward)
    data.to_csv(f"data_plots/{filename}", index=False)
    data.plot(x='step', y='episode_reward', title=filename)
    plt.savefig(f"results_plots/{filename}")
    plt.close()


def scaling_symlog(state):
    state_symlog = np.sign(state) * np.log(np.abs(state)  + 1)
    return state_symlog

def environment_reset(environment):
    try:
        return environment.reset()
    except (EnvironmentError , GripperError) as error:
        error_message = f"Failed to reset with message: {error}"
        logging.error(error_message)
        if handle_gripper_error_home(environment, error_message):
            return environment_reset(environment)  # might keep looping if it keep having issues
        else:
            environment.gripper.close()
    
def read_slack():
    message = slack_bot.get_message("cares-chat-bot")
    
    if message is not None:
        message = message.split(",") 
    else:
        return None

    gripper_id = 9
    if message[0] == str(gripper_id):
        return message[1]
    return None

def handle_gripper_error_home(environment, error_message):
    warning_message = f"Error handling has been initiated because of: {error_message}. Attempting to solve by home sequence."
    logging.warning(warning_message)
    slack_bot.post_message("#bot_terminal", warning_message)
    
    try :
        environment.gripper.wiggle_home()
        return True
    except (EnvironmentError , GripperError):
        warning_message = f"Auto wiggle fix failed, going to final handler"
        logging.warning(warning_message)
        slack_bot.post_message("#bot_terminal", warning_message)
        return handle_gripper_error(environment, error_message)
    
    

def handle_gripper_error(environment, error_message):
    logging.error(f"Error handling has been initiated because of: {error_message}")
    help_message = "Please fix the gripper and press | c to try again | x to quit | w to wiggle:"
    logging.error(help_message)
    slack_bot.post_message("#cares-chat-bot", f"{error_message}, {help_message}")
    
    while True:
        value, timed_out = timedInput(timeout=10)
        if timed_out:
            value = read_slack()

        if value == 'c':
            logging.info("Gripper Fixed continuing onwards")
            return True
        elif value == 'x':
            logging.info("Giving up correcting gripper")
            return False
        elif value == "reboot" or value == "r":
            try:
                logging.info("Rebooting servos")
                environment.gripper.reboot()
            except (EnvironmentError , GripperError):
                warning_message = "Your commanded reboot failed, try again"
                logging.warning(warning_message)
                slack_bot.post_message("#bot_terminal", warning_message)
                return True # stay in loop and try again
            return True
        elif value  == "w":
            try:
                environment.gripper.wiggle_home()
            except (EnvironmentError , GripperError):
                warning_message = "Your commanded wiggle home failed, try again"
                logging.warning(warning_message)
                slack_bot.post_message("#bot_terminal", warning_message)
                return True
            return True
        elif value  == "w2":
            try:
                environment.gripper.move(environment.gripper.max_values)
            except (EnvironmentError , GripperError):
                warning_message = "Your commanded wiggle home 2 failed, try again"
                logging.warning(warning_message)
                slack_bot.post_message("#bot_terminal", warning_message)
                return True
            return True

def parse_args():
    parser = ArgumentParser()
    file_path = Path(__file__).parent.resolve()
    
    parser.add_argument("--learning_config", type=str, default=f"{file_path}/config/learning_config.json")
    parser.add_argument("--env_config",      type=str, default=f"{file_path}/config/env_9DOF_velocity_config.json")  # id 2 for robot left
    parser.add_argument("--gripper_config",  type=str, default=f"{file_path}/config/gripper_9DOF_config.json")
    return parser.parse_args()

def main():

    args = parse_args()
    env_config      = pydantic.parse_file_as(path=args.env_config,      type_=EnvironmentConfig)
    gripper_config  = pydantic.parse_file_as(path=args.gripper_config,  type_=GripperConfig)
    learning_config = pydantic.parse_file_as(path=args.learning_config, type_=LearningConfig)
    create_directories()

    if env_config.env_type == 0:
        environment = RotationEnvironment(env_config, gripper_config)
    elif env_config.env_type == 1:
        environment = TranslationEnvironment(env_config, gripper_config)

    logging.info("Resetting Environment")
    #wrap
    state = environment_reset(environment)
    logging.info(f"State: {state}")
    slack_bot.post_message("#bot_terminal", f"Reset Terminal. \nState: {state}")

    observation_size = len(state)# This wont work for multi-dimension arrays
    action_num       = gripper_config.num_motors
    message = f"Observation Space: {observation_size} Action Space: {action_num}"
    logging.info(message)
    slack_bot.post_message("#bot_terminal", message)

    logging.info("Setting up Seeds")
    torch.manual_seed(learning_config.seed)
    np.random.seed(learning_config.seed)
    random.seed(learning_config.seed)

    logging.info("Setting up Network")
    actor  = Actor(observation_size, action_num, learning_config.actor_lr)
    critic = Critic(observation_size, action_num, learning_config.critic_lr)


    logging.info("Setting up Memory")
    memory = MemoryBuffer(learning_config.buffer_capacity)

    logging.info("Setting RL Algorithm")
    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=learning_config.gamma,
        tau=learning_config.tau,
        action_num=action_num,
        device=DEVICE,
    )

    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_name = f"{date_time_str}_RobotId{gripper_config.gripper_id}_EnvType{env_config.env_type}_ObsType{env_config.object_type}_Seed{learning_config.seed}_{str(agent)[40:43]}"

    logging.info("Starting Training Loop")
    train(environment, agent, memory, learning_config, file_name)

    logging.info("Starting Evaluation Loop")
    #evaluation(environment, agent, file_name)


if __name__ == '__main__':
    main()
