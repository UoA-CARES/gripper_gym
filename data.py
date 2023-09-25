import pickle
import numpy as np
## check data
env_d4rl_name = 'grippertranslation-medium-v2'
dataset_path = f'data/{env_d4rl_name}.pkl'
from decision_transformer.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
# load dataset
with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

min_len = 10**4
states = []
# print(len(trajectories[0]['observations'][0]))
# print(len(trajectories[0]['next_observations'][0]))
# print(len(trajectories[0]['actions']))
# print((trajectories[0]['rewards']))
# print(len(trajectories[0]['terminals']))
print(trajectories)
for traj in trajectories:
    min_len = min(min_len, traj['observations'].shape[0])
    states.append(traj['observations'])

print(states)
# used for input normalization
states = np.concatenate(states, axis=0)
state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

print(dataset_path)
print("num of trajectories in dataset: ", len(trajectories))
print("minimum trajectory length in dataset: ", min_len)
print("state mean: ", state_mean.tolist())
print("state std: ", state_std.tolist())


## check if info is correct
print("is state mean info correct: ", state_mean.tolist() == D4RL_DATASET_STATS[env_d4rl_name]['state_mean'])
print("is state std info correct: ", state_std.tolist() == D4RL_DATASET_STATS[env_d4rl_name]['state_std'])
     
