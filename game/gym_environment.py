# game/gym_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .game_manager import Game
from . import config

class SpaceInvadersEnv(gym.Env):
    def __init__(self):
        super(SpaceInvadersEnv, self).__init__()

        self.action_space = spaces.Discrete(config.NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, 3), 
                                            dtype=np.uint8)
        
        # Initialize game with headless and fast mode
        self.game = Game(silent_mode=True, ai_training_mode=False, headless_worker_mode=False)
        self.fast_mode = True

    def step(self, action):
        # Retrieve the current observation, reward, done status, and info
        observation, reward, done, info = self.game.step_ai(action)
                
        return observation, reward, done, info

    def render(self, mode="rgb_array"):
       observation = self.game._get_observation_for_ai()
       return observation

    def reset(self):
       observation = self.game.reset_for_ai()
       return observation

    def close(self):
        self.game.close()