# utils/agent_factory.py
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from agents.genetic_agent import GeneticAgent
from agents.neat_agent import NEATAgent # Ensure this is correctly imported

def create_agent(agent_name, action_size, obs_shape, hparams=None, mode='train'):
    """
    Factory function to create an agent instance.
    `hparams` is a dictionary of hyperparameters, possibly from a config file.
    `mode` can be 'train', 'test', 'evaluate' to adjust default params (e.g., exploration).
    """
    hparams = hparams if hparams is not None else {}
    # print(f"  Factory: Creating {agent_name} in mode '{mode}' with hparams: {hparams if hparams else 'agent internal defaults'}")

    if agent_name == 'random':
        return RandomAgent(action_size, obs_shape)
    
    # NN-based agents
    if agent_name == 'dqn':
        effective_hparams = hparams.copy()
        if mode != 'train': 
            effective_hparams['eps_start'] = 0.0
            effective_hparams['eps_end'] = 0.0
        return DQNAgent(action_size, obs_shape, **effective_hparams)
    
    elif agent_name == 'a2c':
        effective_hparams = hparams.copy()
        if mode != 'train':
            effective_hparams['entropy_coef'] = 0.0
        return A2CAgent(action_size, obs_shape, **effective_hparams)

    elif agent_name == 'ppo':
        effective_hparams = hparams.copy()
        if mode != 'train':
            effective_hparams['entropy_coef'] = 0.0
        return PPOAgent(action_size, obs_shape, **effective_hparams)

    # Population-based agents
    elif agent_name == 'genetic':
        effective_hparams = hparams.copy()
        if mode != 'train': 
            effective_hparams['population_size'] = 1
            effective_hparams['mutation_rate'] = 0.0
            effective_hparams['num_elites'] = 1
        return GeneticAgent(action_size, obs_shape, **effective_hparams)

    elif agent_name == 'neat':
        effective_hparams = hparams.copy()
        if mode != 'train':
            effective_hparams['population_size'] = 1 
            # For NEAT testing, other evolutionary params might be irrelevant if just loading a genome
        return NEATAgent(action_size, obs_shape, **effective_hparams)

    else:
        raise ValueError(f"Unknown agent type requested in factory: {agent_name}")