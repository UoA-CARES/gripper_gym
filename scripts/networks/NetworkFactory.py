from cares_reinforcement_learning.algorithm.policy import TD3, SAC, DDPG

def create_SAC(observation_size, action_num, learning_config, DEVICE):
    from networks.SAC import Actor
    from networks.SAC import Critic

    actor  = Actor(observation_size, action_num, learning_config.actor_lr)
    critic = Critic(observation_size, action_num, learning_config.critic_lr)

    agent = SAC(
        actor_network=actor,
        critic_network=critic,
        gamma=learning_config.gamma,
        tau=learning_config.tau,
        action_num=action_num,
        device=DEVICE,
    )
    
    return agent

def create_DDPG(observation_size, action_num, learning_config, DEVICE):
    from networks.DDPG import Actor
    from networks.DDPG import Critic

    actor  = Actor(observation_size, action_num, learning_config.actor_lr)
    critic = Critic(observation_size, action_num, learning_config.critic_lr)

    agent = DDPG(
        actor_network=actor,
        critic_network=critic,
        gamma=learning_config.gamma,
        tau=learning_config.tau,
        action_num=action_num,
        device=DEVICE,
    )
    
    return agent

def create_TD3(observation_size, action_num, learning_config, DEVICE):
    from networks.TD3 import Actor
    from networks.TD3 import Critic

    actor  = Actor(observation_size, action_num, learning_config.actor_lr)
    critic = Critic(observation_size, action_num, learning_config.critic_lr)

    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=learning_config.gamma,
        tau=learning_config.tau,
        action_num=action_num,
        device=DEVICE,
    )
    
    return agent

class NetworkFactory:
    def create_network(self, algorithm, observation_size, action_num, learning_config, DEVICE):
        if algorithm == "TD3":
            return create_TD3(observation_size, action_num, learning_config, DEVICE)
        elif algorithm == "DDPG":
            return create_DDPG(observation_size, action_num, learning_config, DEVICE)
        elif algorithm == "SAC":
            return create_SAC(observation_size, action_num, learning_config, DEVICE)
        
        raise ValueError(f"Unknown algorithm: {algorithm}")