# agents/a2c_agent.py
import os
import time
import json
import gymnasium as gym
from gymnasium.wrappers import RecordVideo # RecordEpisodeStatistics is handled by Monitor typically
import numpy as np
import torch # SB3 uses PyTorch

try:
    from stable_baselines3 import A2C as SB3_A2C
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack
    # Using SB3's Monitor wrapper for VecEnv, or RecordEpisodeStatistics for single env.
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.atari_wrappers import AtariWrapper # For single env recording
    from stable_baselines3.common.callbacks import BaseCallback # For custom logging if needed
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 or its dependencies not found. A2CAgent will not be available.")
    # Dummy classes for type hinting if SB3 is not available
    class SB3_A2C: pass
    class VecFrameStack: pass
    class BaseCallback: pass

from .agent import Agent # Your base Agent class
from .dqn_agent import PrintCallback # Re-use the PrintCallback from DQNAgent for logging

class A2CAgent(Agent):
    """
    An Advantage Actor-Critic (A2C) Agent using Stable Baselines 3.
    """
    _default_sb3_a2c_hparams = {
        "learning_rate": 0.0007, # Often 7e-4
        "n_steps": 5,           # Number of steps to run for each environment per update
        "gamma": 0.99,
        "gae_lambda": 1.0,      # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        "ent_coef": 0.0,        # Entropy coefficient for the loss calculation
        "vf_coef": 0.5,         # Value function coefficient for the loss calculation
        "max_grad_norm": 0.5,   # The maximum value for the gradient clipping
        "use_rms_prop": True,   # Whether to use RMSprop (default) or Adam
        "policy_kwargs": None,  # e.g., dict(net_arch=[dict(pi=[64], vf=[64])])
        "tensorboard_log": None,
        "verbose": 0,
        "device": "auto"
    }
    _sb3_env_params = { # Shared with DQN, can be refactored to a common place if many SB3 agents
        "n_envs": 16, # A2C benefits from multiple environments
        "seed": None,
        "wrapper_kwargs": dict(clip_rewards=True, episodic_life=True, fire_on_reset=True),
        "n_stack": 4,
    }
    DEFAULT_ENV_ID = "SpaceInvadersNoFrameskip-v4"

    def __init__(self, env_id, hparams, mode, models_dir_for_agent, gifs_dir_for_agent):
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines 3 is not available. Please install it to use A2CAgent.")

        super().__init__(env_id if env_id else self.DEFAULT_ENV_ID,
                         hparams, mode, models_dir_for_agent, gifs_dir_for_agent)
        
        self.merged_hparams = self._default_sb3_a2c_hparams.copy()
        self.merged_hparams.update(self.hparams)

        self.env_creation_params = self._sb3_env_params.copy()
        if "n_envs" in self.merged_hparams: self.env_creation_params["n_envs"] = self.merged_hparams.pop("n_envs")
        if "seed" in self.merged_hparams: self.env_creation_params["seed"] = self.merged_hparams.pop("seed")
        if "n_stack" in self.merged_hparams: self.env_creation_params["n_stack"] = self.merged_hparams.pop("n_stack")
        
        if self.merged_hparams.get("tensorboard_log") is True:
            self.merged_hparams["tensorboard_log"] = os.path.join(self.models_dir, "tensorboard_logs", "a2c")
        elif isinstance(self.merged_hparams.get("tensorboard_log"), str):
            self.merged_hparams["tensorboard_log"] = os.path.join(self.models_dir, self.merged_hparams["tensorboard_log"])

        try:
            temp_env = make_atari_env(self.env_id, n_envs=1, wrapper_kwargs=self.env_creation_params["wrapper_kwargs"])
            temp_env = VecFrameStack(temp_env, n_stack=self.env_creation_params["n_stack"])
            self.action_size = temp_env.action_space.n
            self.observation_shape = temp_env.observation_space.shape
            print(f"  A2C Agent: Env '{self.env_id}', Action Size: {self.action_size}, Obs Shape: {self.observation_shape}")
            temp_env.close()
            del temp_env
        except Exception as e:
            print(f"  A2C Agent: Could not query temp env for spaces: {e}")
            self.action_size = 6
            self.observation_shape = (self.env_creation_params["n_stack"], 84, 84)

    def get_model_file_extension(self):
        return ".zip"

    def _create_env(self, for_training=True, n_envs_override=None):
        env_seed = self.env_creation_params["seed"]
        if env_seed is None and for_training: env_seed = int(time.time())
        elif env_seed is None and not for_training: env_seed = 42
        
        n_actual_envs = n_envs_override if n_envs_override is not None else self.env_creation_params["n_envs"]
        print(f"  Creating SB3 VecEnv for A2C: {self.env_id}, N_Envs={n_actual_envs}, Seed={env_seed}, N_Stack={self.env_creation_params['n_stack']}")
        vec_env = make_atari_env(
            self.env_id, n_envs=n_actual_envs, seed=env_seed,
            wrapper_kwargs=self.env_creation_params["wrapper_kwargs"]
        )
        vec_env = VecFrameStack(vec_env, n_stack=self.env_creation_params["n_stack"])
        return vec_env

    def train(self, episodes, max_steps_per_episode, render_mode_str,
              path_to_load_model, force_new_training_if_model_exists,
              save_interval_eps, print_interval_steps, **kwargs):
        print(f"\n--- A2C Training Started ---")
        # ... (similar path logic as DQNAgent.train for model_save_path) ...
        self.train_env = self._create_env(for_training=True)

        self.model_save_path = None # Initialize
        if path_to_load_model and os.path.exists(path_to_load_model) and not force_new_training_if_model_exists:
            self.model_save_path = path_to_load_model
            print(f"  Attempting to load and continue training model: {os.path.basename(self.model_save_path)}")
            try:
                self.model = SB3_A2C.load(self.model_save_path, env=self.train_env, device=self.merged_hparams["device"])
                print(f"  A2C Model loaded successfully from {os.path.basename(self.model_save_path)}.")
            except Exception as e:
                print(f"  Failed to load A2C model: {e}. Creating a new model.")
                path_to_load_model = None 
                self.model_save_path = self.get_next_version_save_path("a2c")
                self.model = SB3_A2C("CnnPolicy", self.train_env, **self.merged_hparams)
        else:
            if force_new_training_if_model_exists and path_to_load_model and os.path.exists(path_to_load_model):
                self.model_save_path = self.get_next_version_save_path("a2c")
            elif path_to_load_model and not os.path.exists(path_to_load_model):
                self.model_save_path = self.get_next_version_save_path("a2c")
            else:
                self.model_save_path = self.get_model_save_path_for_agent("a2c")
                if os.path.exists(self.model_save_path) and not force_new_training_if_model_exists:
                     self.model_save_path = self.get_next_version_save_path("a2c")
            print(f"  Creating new A2C model. It will be saved to: {os.path.basename(self.model_save_path)}")
            self.model = SB3_A2C("CnnPolicy", self.train_env, **self.merged_hparams)

        total_timesteps_estimate = episodes * max_steps_per_episode * self.env_creation_params["n_envs"] # A2C is on-policy, more steps needed
        print(f"  Target total timesteps for SB3 A2C learn(): ~{total_timesteps_estimate} (n_envs={self.env_creation_params['n_envs']})")

        callbacks = []
        if print_interval_steps > 0:
            callbacks.append(PrintCallback(print_interval_steps=print_interval_steps))
        if save_interval_eps > 0:
             print(f"  Note: save_interval_eps ({save_interval_eps}) for SB3 A2C is best handled with a custom CheckpointCallback. Model will save at end.")

        try:
            self.model.learn(
                total_timesteps=total_timesteps_estimate,
                callback=callbacks if callbacks else None,
                log_interval=1, # A2C usually logs per rollout (n_steps * n_envs)
                reset_num_timesteps=(path_to_load_model is None)
            )
            print(f"  A2C training loop finished.")
            self.save(self.model_save_path)
        except KeyboardInterrupt:
            print("\n  A2C Training interrupted. Saving current model...")
            self.save(self.model_save_path)
        except Exception as e:
            print(f"  An error occurred during A2C training: {e}")
            import traceback; traceback.print_exc()
            self.save(self.model_save_path)
        finally:
            if self.train_env: self.train_env.close()
            print(f"--- A2C Training Ended ---")

    def test(self, model_path_to_load, episodes, max_steps_per_episode,
             render_during_test, record_video_flag, video_fps, **kwargs):
        print(f"\n--- A2C Testing ---")
        if not model_path_to_load or not os.path.exists(model_path_to_load):
            print(f"  Error: Model path '{model_path_to_load}' not found. Cannot test A2C.")
            if self.mode == 'test' and model_path_to_load is None:
                print(f"  Attempting to test with a new, untrained A2C model instance.")
                temp_eval_env = self._create_env(for_training=False, n_envs_override=1)
                self.model = SB3_A2C("CnnPolicy", temp_eval_env, **self.merged_hparams)
                temp_eval_env.close()
                model_path_to_load = "Untrained A2C Model"
            else:
                return

        # Single environment for rendering and manual episode control
        eval_env = gym.make(self.env_id, render_mode="human" if render_during_test else "rgb_array", full_action_space=False)
        eval_env = AtariWrapper(eval_env, clip_reward=False)
        eval_env = gym.wrappers.FrameStack(eval_env, self.env_creation_params["n_stack"])
        eval_env = Monitor(eval_env) # For episode stats

        if model_path_to_load != "Untrained A2C Model":
            print(f"  Loading A2C model for testing: {os.path.basename(model_path_to_load)}")
            self.model = SB3_A2C.load(model_path_to_load, env=None, device=self.merged_hparams["device"])
        
        all_rewards = []
        for i in range(episodes):
            obs, info = eval_env.reset()
            terminated, truncated = False, False
            episode_reward = 0
            current_steps = 0
            while not (terminated or truncated) and current_steps < max_steps_per_episode:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                current_steps += 1
                if render_during_test and eval_env.render_mode == "human": eval_env.render()
            actual_ep_reward = info.get('episode', {}).get('r', episode_reward)
            all_rewards.append(actual_ep_reward)
            print(f"  A2C Test Episode {i+1}/{episodes} - Score: {actual_ep_reward:.0f}, Steps: {current_steps}")
        eval_env.close()

        avg_reward = np.mean(all_rewards) if all_rewards else 0
        std_reward = np.std(all_rewards) if all_rewards else 0
        print(f"\n  A2C Test Summary: Avg Score: {avg_reward:.2f} +/- {std_reward:.2f}")

        if record_video_flag and model_path_to_load != "Untrained A2C Model":
            print(f"\n  Recording A2C representative run (video)...")
            video_record_env = gym.make(self.env_id, render_mode="rgb_array", full_action_space=False)
            video_record_env = AtariWrapper(video_record_env, clip_reward=False)
            video_record_env = gym.wrappers.FrameStack(video_record_env, self.env_creation_params["n_stack"])
            ts = time.strftime("%Y%m%d_%H%M%S")
            video_folder = os.path.join(self.gifs_dir if self.gifs_dir else "videos", f"a2c_test_{ts}")
            video_env_instance = RecordVideo(
                video_record_env, video_folder=video_folder,
                name_prefix=f"a2c_run_{os.path.basename(model_path_to_load).replace(self.get_model_file_extension(),'')}",
                episode_trigger=lambda ep_id: ep_id == 0, video_length=max_steps_per_episode, fps=video_fps
            )
            try:
                obs, info = video_env_instance.reset()
                terminated, truncated = False, False; vid_ep_reward = 0; vid_steps = 0
                while not (terminated or truncated) and vid_steps < max_steps_per_episode:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = video_env_instance.step(action)
                    vid_ep_reward += reward; vid_steps += 1
                print(f"  A2C Video recorded. Score: {vid_ep_reward:.0f}. Saved in: {video_folder}")
            except Exception as e_vid: print(f"  Error during A2C video recording: {e_vid}")
            finally:
                if video_env_instance: video_env_instance.close()
        print(f"--- A2C Testing Ended ---")

    def evaluate(self, model_path_to_load, episodes, max_steps_per_episode, **kwargs):
        print(f"\n--- A2C Evaluation ---")
        if not model_path_to_load or not os.path.exists(model_path_to_load):
            print(f"  Error: Model path '{model_path_to_load}' not found for A2C. Cannot evaluate.")
            return {}
        
        eval_vec_env = self._create_env(for_training=False, n_envs_override=1) # SB3 evaluate_policy needs VecEnv
        results = {}
        try:
            self.model = SB3_A2C.load(model_path_to_load, env=eval_vec_env, device=self.merged_hparams["device"])
            print(f"  A2C Model loaded for evaluation: {os.path.basename(model_path_to_load)}")
            
            # Use evaluate_policy for a quick summary
            # mean_reward_sb3, std_reward_sb3 = evaluate_policy(
            #     self.model, eval_vec_env, n_eval_episodes=episodes, deterministic=True, render=False, warn=False
            # )

            # Manual loop for detailed stats
            all_ep_rewards, all_ep_steps = [], []
            print(f"  Running manual loop for detailed A2C stats over {episodes} episodes...")
            for i in range(episodes):
                obs = eval_vec_env.reset()
                terminated, truncated = np.array([False]), np.array([False])
                episode_reward, episode_steps = 0, 0
                while not (terminated[0] or truncated[0]) and episode_steps < max_steps_per_episode:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_vec_env.step(action)
                    episode_reward += reward[0]
                    episode_steps += 1
                actual_ep_reward = info[0].get('episode', {}).get('r', episode_reward)
                actual_ep_steps = info[0].get('episode', {}).get('l', episode_steps)
                all_ep_rewards.append(actual_ep_reward)
                all_ep_steps.append(actual_ep_steps)
                if (i + 1) % max(1, episodes // 5) == 0:
                    print(f"    A2C Eval Ep {i+1}/{episodes}: Score={actual_ep_reward:.0f}, Steps={actual_ep_steps}")

            avg_s, std_s = (np.mean(all_ep_rewards), np.std(all_ep_rewards)) if all_ep_rewards else (0,0)
            min_s, max_s = (np.min(all_ep_rewards), np.max(all_ep_rewards)) if all_ep_rewards else (0,0)
            avg_st = np.mean(all_ep_steps) if all_ep_steps else 0
            
            # print(f"  A2C Eval (evaluate_policy): Mean Score: {mean_reward_sb3:.2f} +/- {std_reward_sb3:.2f}")
            print(f"  A2C Eval (manual loop): Avg Score: {avg_s:.2f} +/- {std_s:.2f}")
            print(f"    Min: {min_s:.2f}, Max: {max_s:.2f}, Avg Steps: {avg_st:.1f}")

            results = {
                "num_episodes_eval": episodes, "avg_score": round(avg_s, 2),
                "std_dev_score": round(std_s, 2), "min_score": round(min_s, 2),
                "max_score": round(max_s, 2), "avg_steps": round(avg_st, 1)
            }
        except Exception as e:
            print(f"  Error during A2C evaluation: {e}")
            import traceback; traceback.print_exc()
        finally:
            if eval_vec_env: eval_vec_env.close()
        print(f"--- A2C Evaluation Ended ---")
        return results

    def choose_action(self, observation, deterministic=False):
        if self.model is None:
            if self.action_size is None: return 0
            return np.random.choice(self.action_size)
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path):
        if self.model and path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.model.save(path)
                print(f"  A2C Agent (SB3) model saved to {os.path.basename(path)}")
            except Exception as e: print(f"  Error saving A2C model to {path}: {e}")
        elif not self.model: print("  A2C save attempt: model not initialized.")
        elif not path: print("  A2C save attempt: no path provided.")

    def load(self, path):
        if path and os.path.exists(path):
            try:
                print(f"  A2C Agent (SB3) loading model from {os.path.basename(path)}.")
                # temp_env_for_load = self._create_env(for_training=False, n_envs_override=1)
                # self.model = SB3_A2C.load(path, env=temp_env_for_load, device=self.merged_hparams["device"])
                # A2C can often load without an env if the policy is standard CnnPolicy
                self.model = SB3_A2C.load(path, device=self.merged_hparams["device"])

                print(f"  A2C Model loaded successfully.")
                # if self.model.env: # Update spaces if env was part of loaded model
                #     self.action_size = self.model.env.action_space.n
                #     self.observation_shape = self.model.env.observation_space.shape
                # temp_env_for_load.close()
            except Exception as e:
                print(f"  Error loading A2C model from {path}: {e}")
                import traceback; traceback.print_exc()
                self.model = None
        elif not path: print("  A2C load: path is None.")
        else: print(f"  A2C load: path does not exist: {path}")