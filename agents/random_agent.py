# agents/random_agent.py
import random
from .agent import Agent

class RandomAgent(Agent):
    def __init__(self, action_size, observation_shape=None):
        super().__init__(action_size, observation_shape)

    def choose_action(self, observation): # 'observation' is unused
        """Chooses an action randomly."""
        return random.randint(0, self.action_size - 1)

    def learn(self, *args, **kwargs):
        """Random agent doesn't learn."""
        pass

    # Inherits set_eval_mode, save, load from Agent