# agents/agent.py
import os
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, env_id, hparams, mode, models_dir_for_agent=None, gifs_dir_for_agent=None):
        self.env_id = env_id
        self.hparams = hparams if hparams is not None else {}
        self.mode = mode # 'train', 'test', 'evaluate'
        self.models_dir = models_dir_for_agent
        self.gifs_dir = gifs_dir_for_agent
        
        # Common attributes agents might need, initialized here or in child
        self.model = None # The core learning model (e.g., PyTorch nn.Module, SB3 model)
        self.action_size = None # To be determined by querying env in child
        self.observation_shape = None # To be determined by querying env in child

    @abstractmethod
    def train(self, episodes, max_steps_per_episode, render_mode_str,
              path_to_load_model, force_new_training_if_model_exists,
              save_interval_eps, print_interval_steps, **kwargs):
        pass

    @abstractmethod
    def test(self, model_path_to_load, episodes, max_steps_per_episode,
             render_during_test, record_video_flag, video_fps, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, model_path_to_load, episodes, max_steps_per_episode, **kwargs):
        """Should return a dictionary of metrics."""
        pass

    @abstractmethod
    def choose_action(self, observation, deterministic=False):
        pass

    @abstractmethod
    def save(self, path):
        """Saves the agent's model/state to path."""
        pass

    @abstractmethod
    def load(self, path):
        """Loads the agent's model/state from path."""
        pass
    
    def get_model_save_path_for_agent(self, agent_name_in_main_config, version_suffix=None):
        """ Helper for agents to construct their save paths consistently. """
        if not self.models_dir:
            print("Warning: Models directory not provided to agent.")
            return None # Or a default local path

        # Using the agent_name from main.py's config for filename consistency
        base_filename = f"{agent_name_in_main_config}_spaceinvaders" 
        extension = self.get_model_file_extension() # Agent should define this

        if version_suffix: # e.g., "_v2", "_best"
            return os.path.join(self.models_dir, f"{base_filename}{version_suffix}{extension}")
        else: # Default name or first version
            return os.path.join(self.models_dir, f"{base_filename}{extension}")

    def get_next_version_save_path(self, agent_name_in_main_config):
        """ Finds the next available versioned save path. """
        if not self.models_dir: return None
        
        base_filename = f"{agent_name_in_main_config}_spaceinvaders"
        extension = self.get_model_file_extension()
        
        base_path = os.path.join(self.models_dir, f"{base_filename}{extension}")
        if not os.path.exists(base_path): return base_path
        
        version = 2
        while True:
            versioned_path = os.path.join(self.models_dir, f"{base_filename}_v{version}{extension}")
            if not os.path.exists(versioned_path): return versioned_path
            version += 1

    @abstractmethod
    def get_model_file_extension(self):
        """ Agent must declare its model file extension (e.g., '.zip', '.pth', '.pkl') """
        pass

    # Optional: Common preprocessing if many agents use it
    # def preprocess_observation(self, raw_obs, new_size=(84,84)):
    #     from PIL import Image
    #     import numpy as np
    #     # ... (implementation from your main.py) ...
    #     return processed_obs