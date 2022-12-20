
"""
This is an example script that shows how one uses the cares reinforcement learning package.
To run this specific example, move the file so that it is at the same level as the package root
directory
    -- script.py
    -- summer_reinforcement_learning/
"""

#network
#memory replays 

#TODO: training loop, selecting an action, exploration phase, create environment 
#TODO: make sure data is normalised 
#TODO: figure out a better way to do the max and min (current idea is to have limits initialised with the servo when its initialised)
#TODO: add argument parsers bc the way david explained it made it seem very uself

from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer
from cares_reinforcement_learning.examples.Actor import Actor
from cares_reinforcement_learning.examples.Critic import Critic

from  GripperClass import Gripper
import numpy as np
from argparse import ArgumentParser
import random
#from Servo import Servo
#from Camera import Camera

import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")

BUFFER_CAPACITY = 10

GAMMA = 0.995
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

EPISODE_NUM = 10
BATCH_SIZE = 8

env = Gripper() #--> env.reset, env.move(actions), 

MAX_ACTIONS = np.array([1023, (1023-150), 769, 1023, (1023-168), 802, 1023, (1023-190), 794])
MIN_ACTIONS = np.array([0,(1023-784), 130, 0, (1023-729), 152, 0, (1023-795), 140])

UPPER_LIMIT = np.array([1023, 150, 769, 1023, 168, 802, 1023, 190, 794])
LOWER_LIMIT = np.array([0, 784, 130, 0, 729, 152, 0, 795, 140])


def main():

    observation_size = 10  
    action_num = 9

    #setup the grippers
    

    # TODO: add angle pair limits (maybe check after they have gone through the network)
    # TODO: change this once i change the max min thing in the servo class
    max_actions = MAX_ACTIONS
    min_actions = MIN_ACTIONS

    memory = MemoryBuffer(BUFFER_CAPACITY)

    actor = Actor(observation_size, action_num, ACTOR_LR, max_actions)
    critic_one = Critic(observation_size, action_num, CRITIC_LR)
    critic_two = Critic(observation_size, action_num, CRITIC_LR)

    #TODO: implement the argument parser

    #args   = define_parse_args()

    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)

    td3 = TD3(
        actor_network=actor,
        critic_one=critic_one,
        critic_two=critic_two,
        max_actions=max_actions,
        min_actions=min_actions,
        gamma=GAMMA,
        tau=TAU,
        device=DEVICE
    )

    print(f"Filling Buffer...")

    fill_buffer(memory)

    train(td3, memory)


def train(td3, memory: MemoryBuffer):
    historical_reward = []

    for episode in range(0, EPISODE_NUM):

        state = env.reset()
        episode_reward = 0
        print(state)

        while episode < EPISODE_NUM:  #need to check whether this is right

            # Select an Action
            #td3.actor_net.eval() --> dont need bc we are not using batch norm
            with torch.no_grad():
                #print(state)
                state_tensor = torch.FloatTensor(state) 
                print("Size of the int_list_to_float_tensor: ", state_tensor.size())
                print("Dimensions of the int_list_to_float_tensor: ",state_tensor.ndimension())
                #state_tensor = state_tensor
                state_tensor = state_tensor.to(DEVICE)
                action = td3.forward(state_tensor) #potientially a naming conflict
                action = action.numpy()
            td3.actor_net.train(True)

            #convert actor output to valid integer steps within the max and min
            for i in range(0, len(action)):
                #TODO make sure this is actually doing what I want it to do 
                action[i] = action[i] * (UPPER_LIMIT[i] - LOWER_LIMIT[i]) + LOWER_LIMIT[i]
        
            action = action.astype(int)
            print(action.dtype)
            print(action)
            target_angle = np.random.randint(0, 360)

            next_state, reward, = env.move(action, target_angle)
            memory.add(state, action, reward, False, next_state)
            print(reward)
            experiences = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(experiences)
            print(states)
            for _ in range(0, 10):
                print("learning")
                td3.learn(experiences)

            state = next_state
            episode_reward += reward

            if episode > EPISODE_NUM:
                break

        historical_reward.append(episode_reward)
        print(f"Episode #{episode} Reward {episode_reward}")


def fill_buffer(memory):

    env.setup()
    state = env.reset()
 
    while len(memory.buffer) < memory.buffer.maxlen:
      
            
        # TODO: refactor the code surely i can make it better than this
        action = np.zeros(9)
    
        for i in range(0, len(MAX_ACTIONS)):
            action[i] = np.random.randint(MIN_ACTIONS[i], MAX_ACTIONS[i])

        action = action.astype(int)
        #pick a random target angle
        target_angle = np.random.randint(0, 360)
        #TODO: would be good to have a thing here to add a thing to the memory if the actions terminated
        #TODO: figure out how to incorporate the target angle
        next_state, reward = env.move(action, target_angle)
        
        memory.add(state, action, reward, False, next_state)

        #how full is the buffer?
        print(f"Buffer: {len(memory.buffer)} / {memory.buffer.maxlen}", end='\r')
        

        state = next_state



if __name__ == '__main__':
    main()