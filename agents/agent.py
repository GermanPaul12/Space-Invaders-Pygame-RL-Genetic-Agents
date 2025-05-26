# agents/agent.py
import abc

class Agent(abc.ABC):
    def __init__(self, action_size, observation_shape=None):
        self.action_size = action_size
        self.observation_shape = observation_shape # (e.g., for NN input)

    @abc.abstractmethod
    def choose_action(self, observation):
        """Given an observation, choose an action."""
        pass

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        """Update the agent's policy based on experience."""
        pass

    def save(self, path):
        """Save agent's model/parameters."""
        print(f"Save method not implemented for {self.__class__.__name__}")

    def load(self, path):
        """Load agent's model/parameters."""
        print(f"Load method not implemented for {self.__class__.__name__}")