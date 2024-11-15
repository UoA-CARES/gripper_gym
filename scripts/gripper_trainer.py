import logging
import time
from datetime import datetime

import torch
from rl_zoo.networks.mfrl.tqc.critic_tqc import TQC_Critic
from rl_zoo.networks.mfrl.common.actor import Actor
from rl_zoo.networks.mfrl.sac.critic_sac import SAC_Critic
from rl_zoo.networks.world_models.ensembles import Ensemble_Dyna_One_Reward
from rl_zoo.networks.world_models.deterministic import Single_PNN
import numpy as np

from rl_zoo.agents.mfrl.sac import SAC
from rl_zoo.agents.mfrl.tqc import TQC
from rl_zoo.agents.mbrl import Dyna_SAC_NS

import tools.error_handlers as erh
from cares_lib.dynamixel.Gripper import GripperError
from cares_lib.dynamixel.gripper_configuration import GripperConfig
from rl_zoo.utils import PrioritizedReplayBuffer
from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from configurations import GripperEnvironmentConfig
from environments.environment_factory import EnvironmentFactory

logging.basicConfig(level=logging.INFO)


class GripperTrainer:
    def __init__(
            self,
            env_config: GripperEnvironmentConfig,
            training_config: TrainingConfig,
            alg_config: AlgorithmConfig,
            gripper_config: GripperConfig,
    ) -> None:
        """
        Initializes the GripperTrainer class for training gripper actions in various environments.

        Args:
        env_config (Object): Configuration parameters for the environment.
        gripper_config (Object): Configuration parameters for the gripper.
        alg_config (Object): Configuration parameters for the learning/training process.
        file_path (str): Path to save or retrieve model related files.

        Many of the instances variables are initialised from the provided configurations and are used throughout the training process.
        """

        self.env_config = env_config
        self.train_config = training_config
        self.alg_config = alg_config
        self.gripper_config = gripper_config

        env_factory = EnvironmentFactory()

        # TODO add set_seed to environment
        self.environment = env_factory.create_environment(env_config, gripper_config)

        logging.info("Resetting Environment")
        # will just crash right away if there is an issue but that is fine
        state = self.environment.reset()
        logging.info(f"State: {state}")
        
        # This wont work for multi-dimension arrays - TODO push this to the environment
        observation_size = len(state)
        action_num = gripper_config.num_motors
        logging.info(
            f"Observation Space: {observation_size} Action Space: {action_num}"
        )
        ########################################################################
        algo = alg_config.algorithm
        print(algo)
        actor = Actor(observation_size, action_num)
        self.memory = PrioritizedReplayBuffer()

        if algo == 'SAC':
            critic = SAC_Critic(observation_size, action_num)
            self.agent = SAC(actor, critic, gamma=0.99,
                             tau=0.005, reward_scale=1.0,
                             action_num=action_num,
                             actor_lr=1e-3, critic_lr=1e-3,
                             alpha_lr=0.001, device="cuda")
        if algo == 'tqc':
            critic = TQC_Critic(observation_size=observation_size,
                                num_actions=action_num,
                                num_critics=5,
                                num_quantiles=25)
            self.agent = TQC(actor, critic, gamma=0.99,
                             tau=0.005, action_num=action_num,
                             top_quantiles_to_drop=2,
                             actor_lr=3e-4, critic_lr=3e-4,
                             alpha_lr=0.001, device="cuda")
        if algo == 'DynaSAC':
            self.world_model = Single_PNN(observation_size,
                                                action_num,
                                                device='cuda')
            critic = SAC_Critic(observation_size, action_num)
            self.agent = Dyna_SAC_NS(actor_network=actor,
                                     critic_network=critic,
                                     world_network=self.world_model,
                                     gamma=0.99,
                                     tau=0.005,
                                     action_num=action_num,
                                     actor_lr=1e-3,
                                     critic_lr=1e-3,
                                     alpha_lr=0.001,
                                     num_samples=40,
                                     horizon=1,
                                     device='cuda')

        ########################################################################

        # TODO: reconcile deep file_path dependency

        self.file_path = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-gripper{gripper_config.gripper_id}-{env_config.task}-{alg_config.algorithm}-{training_config.seeds}-{gripper_config.action_type}'

        self.record = Record(
            log_dir=self.file_path,
            algorithm=alg_config.algorithm,
            task=env_config.task,
            plot_frequency=training_config.plot_frequency,
            checkpoint_frequency=training_config.checkpoint_frequency,
            network=self.agent,
        )

        self.record.save_config(env_config, "env_config")
        self.record.save_config(alg_config, "alg_config")
        self.record.save_config(training_config, "training_config")
        self.record.save_config(gripper_config, "gripper_config")

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
                # might keep looping if it keep having issues
                return self.environment_reset()
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
                state = self.environment.reset()
                # Truncated to True to skip the episode
                return state, 0, False, True
            else:
                self.environment.gripper.close()
                self.agent.save_models("error_models", self.file_path)
                exit(1)

    def train(self):
        """
        This method is the main training loop that is called to start training the agent a given environment.
        Trains the agent and save the results in a file and periodically evaluate the agent's performance as well as plotting results.
        Logging and messaging to a specified Slack Channel for monitoring the progress of the training.
        """
        start_time = time.time()

        max_steps_training = self.alg_config.max_steps_training
        max_steps_exploration = self.alg_config.max_steps_exploration
        number_steps_per_evaluation = self.train_config.number_steps_per_evaluation
        number_steps_per_train_policy = self.alg_config.number_steps_per_train_policy
        batch_size = self.alg_config.batch_size
        G = self.alg_config.G

        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        success_counter = 0
        steps_to_success = 0
        
        evaluate = False

        state = self.environment_reset()

        episode_start = time.time()
        for total_step_counter in range(int(max_steps_training)):
            episode_timesteps += 1
            action = self.agent.select_action_from_policy(state)
            # gripper range
            action_env = self.environment.denormalize(action)
            next_state, reward_extrinsic, done, truncated = self.environment_step(
                action_env
            )
            if reward_extrinsic >= self.environment.goal_reward:
                if steps_to_success == 0:
                    steps_to_success = self.environment.step_counter
                success_counter += 1

            env_start_time = time.time()
            self.memory.add(state, action, reward_extrinsic, next_state, done)

            state = next_state
            # Note we only track the extrinsic reward for the episode for proper comparison
            episode_reward += reward_extrinsic

            # Regardless if velocity or position based, train every step
            start_train_time = time.time()
            if (
                total_step_counter >= max_steps_exploration
                and total_step_counter % number_steps_per_train_policy == 0
            ):
                if len(self.memory) == (batch_size + 1):
                    statistics = self.memory.get_statistics()
                    self.agent.set_statistics(statistics)
                for _ in range(G):
                    self.agent.train_policy(self.memory, batch_size)

                for _ in range(int(20)):
                    self.agent.train_world_model(memory=self.memory,
                                                 batch_size=batch_size)

            end_train_time = time.time()
            logging.debug(
                f"Time to run training loop {end_train_time-start_train_time} \n"
            )

            if (total_step_counter + 1) % number_steps_per_evaluation == 0:
                evaluate = True

            if done or truncated:
                if len(self.memory) > batch_size:
                    statistics = self.memory.get_statistics()
                    self.agent.set_statistics(statistics)

                episode_time = time.time() - episode_start
                self.record.log_train(
                    total_steps=total_step_counter + 1,
                    episode=episode_num + 1,
                    episode_steps=episode_timesteps,
                    episode_reward=episode_reward,
                    episode_time=episode_time,
                    success_counter = success_counter,
                    steps_to_success = steps_to_success,
                    display=self.env_config.display,
                )

                if evaluate & (total_step_counter > max_steps_exploration):
                    logging.info("*************--Evaluation Loop--*************")
                    self.evaluation_loop(total_step_counter)
                    evaluate = False
                    logging.info("--------------------------------------------")

                # Reset environment
                state = self.environment_reset()

                episode_timesteps = 0
                episode_reward = 0
                episode_num += 1
                success_counter = 0
                steps_to_success = 0
                episode_start = time.time()

            # Run loop at a fixed frequency
            if self.gripper_config.action_type == "velocity":
                self.dynamic_sleep(env_start_time)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        if self.gripper_config.touch:
            self.environment.Touch.stop()

    def evaluation_loop(self, total_steps, num_eval_steps=None,
                        num_eval_episodes=None):
        """
        Executes an evaluation loop to assess the agent's performance.

        Args:
        total_counter: The total step count leading up to the current evaluation loop.
        file_name: Name of the file where evaluation results will be saved.
        historical_reward_evaluation: Historical rewards list that holds average rewards from previous evaluations.

        The method aims to evaluate the agent's performance by running the environment for a set number of steps and recording the average reward.
        """
        model_errors = 0.0
        eval_steps = 1
        number_eval_episodes = int(self.train_config.number_eval_episodes)
        state = self.environment_reset()
        frame = self.environment.grab_rendered_frame()
        self.record.start_video(total_steps + 1, frame, fps=1)
        for eval_episode_counter in range(number_eval_episodes):
            episode_timesteps = 0
            episode_reward = 0
            episode_num = 0
            success_counter = 0
            steps_to_success = 0
            done = False
            truncated = False
            while not done and not truncated:
                episode_timesteps += 1
                action = self.agent.select_action_from_policy(state,
                                                              evaluation=True)
                action_env = self.environment.denormalize(action)
                next_state, reward, done, truncated = self.environment_step(
                    action_env)

                if self.alg_config.algorithm == 'DynaSAC':
                    # Simple Evaluate the world model
                    tensor_state = torch.FloatTensor(state).to('cuda').unsqueeze(
                        dim=0)
                    tensor_action = torch.FloatTensor(action_env).to(
                        'cuda').unsqueeze(dim=0)
                    pred_next, _, _, _ = self.agent.world_model.pred_next_states(
                        tensor_state, tensor_action)
                    pred_next = pred_next.squeeze().detach().cpu().numpy()
                    pred_next = np.concatenate((pred_next, next_state[-2:]))
                    mse_loss = np.mean((pred_next - state) ** 2) ** 0.5
                    model_errors += mse_loss
                    eval_steps += 1

                if reward >= self.environment.goal_reward:
                    if steps_to_success == 0:
                        steps_to_success = self.environment.step_counter
                    success_counter += 1
                start_time = time.time()
                # record all eps in the same timestamp video
                frame = self.environment.grab_rendered_frame()
                self.record.log_video(frame)
                episode_reward += reward
                state = next_state
                if done or truncated:
                    self.record.log_eval(
                        total_steps=total_steps + 1,
                        episode=eval_episode_counter + 1,
                        episode_reward=episode_reward,
                        success_counter=success_counter,
                        steps_to_success=steps_to_success,
                        display=self.env_config.display,
                    )
                    state = self.environment_reset()
                    episode_reward = 0
                    episode_timesteps = 0
                    success_counter = 0
                    steps_to_success = 0
                    episode_num += 1
                # Run loop at a fixed frequency
                if self.gripper_config.action_type == "velocity":
                    self.dynamic_sleep(start_time)
        logging.info(f"Model Error: {model_errors / eval_steps}")
        self.record.stop_video()

    def dynamic_sleep(self, env_start):
        process_time = time.time() - env_start
        logging.debug(
            f"Time to process training loop: {process_time}/{self.env_config.step_time_period} secs"
        )

        delay = self.env_config.step_time_period - process_time
        if delay > 0:
            time.sleep(delay)
