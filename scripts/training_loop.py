import logging
logging.basicConfig(level=logging.INFO)

import os
import pydantic
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from argparse import ArgumentParser

import torch
import random
import numpy as np

from environments.RotationEnvironment import RotationEnvironment
from environments.TranslationEnvironment import TranslationEnvironment
from configurations import LearningConfig, EnvironmentConfig, GripperConfig


from networks import Actor
from networks import Critic
from cares_reinforcement_learning.algorithm import TD3
from cares_reinforcement_learning.util import MemoryBuffer


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logging.info("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    logging.info("Working with CPU")


def evaluation(environment, agent, file_name):
    agent.load_models(filename=file_name)

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    max_steps_evaluation        = 100
    episode_horizont_evaluation = 20

    state = environment.reset()

    for total_step_counter in range(max_steps_evaluation):
        episode_timesteps += 1

        action = agent.select_action_from_policy(state, evaluation=True)  # algorithm range [-1, 1]
        action_env = environment.denormalize(action)  # gripper range
        next_state, reward, done, truncated = environment.step(action_env)
        logging.info(f"Reward of this step:{reward}")
        state = next_state
        episode_reward += reward

        if done is True or episode_timesteps >= episode_horizont_evaluation:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            # Reset environment
            state =  environment.reset()

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

    state = environment.reset()
    for total_step_counter in range(int(learning_config.max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < learning_config.max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{learning_config.max_steps_exploration}")
            action_env = environment.sample_action() # gripper range
            action     = environment.normalize(action_env) # algorithm range [-1, 1]
        else:
            noise_scale *= noise_decay
            noise_scale = max(min_noise, noise_scale)
            logging.info(f"Noise Scale:{noise_scale}")

            logging.info(f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n")
            action = agent.select_action_from_policy(state, noise_scale=noise_scale)  # algorithm range [-1, 1]
            action_env = environment.denormalize(action)  # gripper range

        next_state, reward, done, truncated = environment.step(action_env)
        logging.info(f"Reward of this step:{reward}")

        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state

        episode_reward += reward

        if total_step_counter >= learning_config.max_steps_exploration:
            for _ in range(learning_config.G):
                experiences = memory.sample(learning_config.batch_size)
                agent.train_policy(experiences)

        if done is True or episode_timesteps >= learning_config.episode_horizont:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state =  environment.reset()

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


def parse_args():
    parser = ArgumentParser()
    file_path = Path(__file__).parent.resolve()
    
    parser.add_argument("--learning_config", type=str, default=f"{file_path}/config/learning_config_ID1.json")
    parser.add_argument("--env_config",      type=str, default=f"{file_path}/config/env_4DOF_config_ID1.json")  # id 2 for robot left
    parser.add_argument("--gripper_config",  type=str, default=f"{file_path}/config/gripper_4DOF_config_ID1.json")
    return parser.parse_args()

def main():

    args = parse_args()
    env_config      = pydantic.parse_file_as(path=args.env_config,      type_=EnvironmentConfig)
    gripper_config  = pydantic.parse_file_as(path=args.gripper_config,  type_=GripperConfig)
    learning_config = pydantic.parse_file_as(path=args.learning_config, type_=LearningConfig)

    if env_config.env_type == 0:
        environment = RotationEnvironment(env_config, gripper_config)
    elif env_config.env_type == 1:
        environment = TranslationEnvironment(env_config, gripper_config)


    logging.info("Resetting Environment")
    state = environment.reset()
    logging.info(f"State: {state}")

    observation_size = len(state)# This wont work for multi-dimension arrays
    action_num       = gripper_config.num_motors
    logging.info(f"Observation Space: {observation_size} Action Space: {action_num}")

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

    file_name = f"Friday_14_RobotId{gripper_config.gripper_id}_EnvType{env_config.env_type}_ObsType{env_config.object_type}_Seed{learning_config.seed}_{str(agent)[40:43]}"
    create_directories()

    logging.info("Starting Training Loop")
    train(environment, agent, memory, learning_config, file_name)

    logging.info("Starting Evaluation Loop")
    #evaluation(environment, agent, file_name)


if __name__ == '__main__':
    main()
