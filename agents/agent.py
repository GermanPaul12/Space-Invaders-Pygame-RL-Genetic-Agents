# agents/agent.py
import abc

class Agent(abc.ABC):
    def __init__(self, action_size, observation_shape=None):
        self.action_size = action_size
        self.observation_shape = observation_shape
        self.is_evaluating = False # Default to not evaluating (i.e., training mode)

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

    def set_eval_mode(self, is_evaling: bool):
        """Sets the agent's mode to evaluation (True) or training (False)."""
        self.is_evaluating = is_evaling
        # Optional: print a confirmation, can be useful for debugging
        # mode_str = "EVALUATION" if is_evaling else "TRAINING"
        # print(f"  INFO: {self.__class__.__name__} set to {mode_str} mode.")