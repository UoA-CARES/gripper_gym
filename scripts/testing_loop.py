
"""
This is an example script that shows how one uses the cares reinforcement learning package.
To run this specific example, move the file so that it is at the same level as the package root
directory
    -- script.py
    -- summer_reinforcement_learning/
"""

#network
#memory replays 

#TODO: track the error messages so i know when things are breaking


from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer
#TODO: figure out why this isnt working
#from cares_reinforcement_learning.examples.Actor import Actor
#from cares_reinforcement_learning.examples.Critic import Critic

from gripper_environment import Environment
import numpy as np
from argparse import ArgumentParser
import random
import matplotlib.pyplot as plt

#these are just for the networks that should get moved
import torch
import torch.nn as nn
import torch.optim as optim

#if torch.cuda.is_available():
   # DEVICE = torch.device('cuda')
    #print("Working with GPU")

DEVICE = torch.device('cpu')
print("Working with CPU")

#BUFFER_CAPACITY = 10

GAMMA = 0.995
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

#EPISODE_NUM = 10
#BATCH_SIZE = 8  #32 good

MAX_ACTIONS = np.array([900, 750, 750, 900, 750, 750, 900, 750, 750])  #have generalised this to 750 for lower joints for consistency
MIN_ACTIONS = np.array([100, 250, 250, 100, 250, 250, 100, 250, 250]) #have generalised this to 250 for lower joints for consistency

env = Environment()


#need to move these
class Actor(nn.Module):

    def __init__(self, observation_size, num_actions, learning_rate, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.hidden_size = [128, 64, 32]

        self.h_linear_1 = nn.Linear(in_features=observation_size, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=num_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.sigmoid(self.h_linear_4(x))
        return x

class Critic(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Critic, self).__init__()

        self.hidden_size = [128, 64, 32]

        self.Q1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], self.hidden_size[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], 1)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.Q1(x)
        return q1


def main():

    observation_size = 10  

    action_num = 9

    #setup the grippers
    args = parse_args()
    
    # TODO: change this once i change the max min thing in the servo class
    max_actions = MAX_ACTIONS
    min_actions = MIN_ACTIONS

    memory = MemoryBuffer(args.buffer_capacity)

    actor = Actor(observation_size, action_num, ACTOR_LR, max_actions)
    critic_one = Critic(observation_size, action_num, CRITIC_LR)
    critic_two = Critic(observation_size, action_num, CRITIC_LR)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    args = parse_args()

    historical_reward = []

    state = env.gripper.home()
    #to get the array the correct length for the first action I need to  
    state.append(-1)

    for episode in range(0, args.episode_num):

        #map state values to 0 - 1 
        for i in range(0, len(state)):
            state[i] = (state[i])/360
          
        episode_reward = 0 
        Done = False
        action_taken = 0
        target_angle = np.random.randint(0, 360)
        print(f"episode {episode}")
        #print(state)


        while not Done and action_taken < args.action_num: 

            # Select an Action
            #td3.actor_net.eval() --> dont need bc we are not using batch norm???
            with torch.no_grad():
                
                state_tensor = torch.FloatTensor(state) 
                
                state_tensor = state_tensor.to(DEVICE)
                action = td3.forward(state_tensor) #potientially a naming conflict
                action = action.numpy()

            td3.actor_net.train(True)

            #convert actor output to valid integer steps within the max and min
            for i in range(0, len(action)):
                #map 0 - 1 to min - max
                action[i] = (action[i]) * (MAX_ACTIONS[i] - MIN_ACTIONS[i]) + MIN_ACTIONS[i]
            
            action = action.astype(int)

            next_state, reward, terminated, Done = env.step(action, target_angle, action_taken)
            
            memory.add(state, action, reward, next_state, Done)

            experiences = memory.sample(args.batch_size)
            
            for _ in range(0, 10): #can be bigger
                
                td3.learn(experiences)

            action_taken += 1
            print(f"actions taken = {action_taken}")

            state = next_state
            episode_reward += reward

            if terminated:
                print("Episode Terminated")
                historical_reward.append(episode_reward)
                episode += 1

        historical_reward.append(episode_reward)

        if episode % 100:
            f = open("testinglog.txt", "a")
            f.write(f"the current epsiode is {episode}, the number of actions taken was {action_taken}, the reward was {episode_reward} ")
            f.close
        plt.plot(historical_reward)
        print(f"Episode #{episode} Reward {episode_reward}")

        plt.plot(historical_reward)
    plt.xlabel("Episode")
    plt.ylabel("Reward") 
    plt.title("Reward per Episode")     
    xint = []
    locs, labels = plt.xticks()
    for each in locs:
        xint.append(int(each))
    plt.xticks(xint)
    plt.show()


def fill_buffer(memory):

    env.gripper.setup()
    state = env.gripper.home()
 
    while len(memory.buffer) < memory.buffer.maxlen:
      
            
        # TODO: refactor the code surely i can make it better than this
        action = np.zeros(9)
        action_taken = 0 #need it for step but is irrevilant at this point
    
        for i in range(0, len(MAX_ACTIONS)):
            action[i] = np.random.randint(MIN_ACTIONS[i], MAX_ACTIONS[i])

        action = action.astype(int)
        #pick a random target angle
        target_angle = np.random.randint(0, 360)
        #TODO: would be good to have a thing here to add a thing to the memory if the actions terminated
        next_state, reward, terminated, done = env.step(action, target_angle, action_taken)
        print(reward)
        memory.add(state, action, reward, next_state, done)
        #keep track of how full the buffer is 
        print(f"Buffer: {len(memory.buffer)} / {memory.buffer.maxlen}", end='\r')
        state = next_state

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_capacity", type=int, default=100)
    parser.add_argument("--episode_num", type=int, default=100)
    parser.add_argument("--action_num", type=int, default=15)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()