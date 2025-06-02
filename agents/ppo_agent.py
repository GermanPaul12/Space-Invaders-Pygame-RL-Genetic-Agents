# agents/ppo_agent.py
import os
import time
import json
import gymnasium as gym
from gymnasium.wrappers import RecordVideo # RecordEpisodeStatistics is typically handled by Monitor
import numpy as np
import torch # SB3 uses PyTorch

try:
    from stable_baselines3 import PPO as SB3_PPO
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack
    from stable_baselines3.common.monitor import Monitor # For single env episode stats
    from stable_baselines3.common.atari_wrappers import AtariWrapper # For single env recording/testing
    from stable_baselines3.common.callbacks import BaseCallback # For custom logging if needed
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 or its dependencies not found. PPOAgent will not be available.")
    # Dummy classes for type hinting if SB3 is not available
    class SB3_PPO: pass
    class VecFrameStack: pass
    class BaseCallback: pass

from .agent import Agent # Your base Agent class
from .dqn_agent import PrintCallback # Re-use the PrintCallback for logging consistency

class PPOAgent(Agent):
    """
    A Proximal Policy Optimization (PPO) Agent using Stable Baselines 3.
    """
    _default_sb3_ppo_hparams = {
        "learning_rate": 0.000025, # Often 2.5e-4 or 3e-4 for PPO on Atari
        "n_steps": 128,         # Number of steps to run for each environment per update (rollout buffer size = n_steps * n_envs)
        "batch_size": 256,       # Minibatch size for PPO updates (SB3 PPO calculates this based on n_envs and n_steps for full batch, then sub-batches)
                                # For SB3, batch_size is often n_steps * n_envs / nminibatches. Let's set nminibatches.
        "n_epochs": 4,          # Number of epochs when optimizing the surrogate loss
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,      # Clipping parameter, often 0.1 or 0.2
        "ent_coef": 0.01,       # Entropy coefficient
        "vf_coef": 0.5,         # Value function coefficient
        "max_grad_norm": 0.5,
        "policy_kwargs": None,  # e.g., dict(net_arch=[dict(pi=[64], vf=[64])])
        "tensorboard_log": None,
        "verbose": 0,
        "device": "auto"
    }
    # PPO benefits greatly from multiple environments
    _sb3_env_params = {
        "n_envs": 8,
        "seed": None,
        "wrapper_kwargs": dict(clip_rewards=True, episodic_life=True, fire_on_reset=True),
        "n_stack": 4,
    }
    DEFAULT_ENV_ID = "SpaceInvadersNoFrameskip-v4"

    def __init__(self, env_id, hparams, mode, models_dir_for_agent, gifs_dir_for_agent):
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines 3 is not available. Please install it to use PPOAgent.")

        super().__init__(env_id if env_id else self.DEFAULT_ENV_ID,
                         hparams, mode, models_dir_for_agent, gifs_dir_for_agent)
        
        self.merged_hparams = self._default_sb3_ppo_hparams.copy()
        self.merged_hparams.update(self.hparams)

        self.env_creation_params = self._sb3_env_params.copy()
        if "n_envs" in self.merged_hparams: self.env_creation_params["n_envs"] = self.merged_hparams.pop("n_envs")
        if "seed" in self.merged_hparams: self.env_creation_params["seed"] = self.merged_hparams.pop("seed")
        if "n_stack" in self.merged_hparams: self.env_creation_params["n_stack"] = self.merged_hparams.pop("n_stack")
        
        # SB3 PPO's batch_size is typically total rollout buffer size / num_minibatches
        # We will let SB3 handle it by not explicitly setting batch_size unless user overrides.
        # If user provides batch_size, we assume it's the minibatch_size SB3 PPO expects.
        # Default n_steps = 128, n_envs = 8 -> rollout = 1024. With n_epochs=4.
        # Default batch_size in SB3 PPO is 64. So 1024/64 = 16 minibatches.
        # If we want to control minibatches, we can calculate batch_size, but it's usually fine.


        if self.merged_hparams.get("tensorboard_log") is True:
            self.merged_hparams["tensorboard_log"] = os.path.join(self.models_dir, "tensorboard_logs", "ppo")
        elif isinstance(self.merged_hparams.get("tensorboard_log"), str):
            self.merged_hparams["tensorboard_log"] = os.path.join(self.models_dir, self.merged_hparams["tensorboard_log"])

        try:
            temp_env = make_atari_env(self.env_id, n_envs=1, wrapper_kwargs=self.env_creation_params["wrapper_kwargs"])
            temp_env = VecFrameStack(temp_env, n_stack=self.env_creation_params["n_stack"])
            self.action_size = temp_env.action_space.n
            self.observation_shape = temp_env.observation_space.shape
            print(f"  PPO Agent: Env '{self.env_id}', Action Size: {self.action_size}, Obs Shape: {self.observation_shape}")
            temp_env.close()
            del temp_env
        except Exception as e:
            print(f"  PPO Agent: Could not query temp env for spaces: {e}")
            self.action_size = 6
            self.observation_shape = (self.env_creation_params["n_stack"], 84, 84)

    def get_model_file_extension(self):
        return ".zip"

    def _create_env(self, for_training=True, n_envs_override=None):
        env_seed = self.env_creation_params["seed"]
        if env_seed is None and for_training: env_seed = int(time.time())
        elif env_seed is None and not for_training: env_seed = 42
        
        n_actual_envs = n_envs_override if n_envs_override is not None else self.env_creation_params["n_envs"]
        print(f"  Creating SB3 VecEnv for PPO: {self.env_id}, N_Envs={n_actual_envs}, Seed={env_seed}, N_Stack={self.env_creation_params['n_stack']}")
        vec_env = make_atari_env(
            self.env_id, n_envs=n_actual_envs, seed=env_seed,
            wrapper_kwargs=self.env_creation_params["wrapper_kwargs"]
        )
        vec_env = VecFrameStack(vec_env, n_stack=self.env_creation_params["n_stack"])
        return vec_env

    def train(self, episodes, max_steps_per_episode, render_mode_str,
              path_to_load_model, force_new_training_if_model_exists,
              save_interval_eps, print_interval_steps, **kwargs):
        print(f"\n--- PPO Training Started ---")
        self.train_env = self._create_env(for_training=True)

        self.model_save_path = None # Initialize
        if path_to_load_model and os.path.exists(path_to_load_model) and not force_new_training_if_model_exists:
            self.model_save_path = path_to_load_model
            print(f"  Attempting to load and continue training PPO model: {os.path.basename(self.model_save_path)}")
            try:
                self.model = SB3_PPO.load(self.model_save_path, env=self.train_env, device=self.merged_hparams["device"])
                print(f"  PPO Model loaded successfully from {os.path.basename(self.model_save_path)}.")
            except Exception as e:
                print(f"  Failed to load PPO model: {e}. Creating a new model.")
                path_to_load_model = None 
                self.model_save_path = self.get_next_version_save_path("ppo")
                self.model = SB3_PPO("CnnPolicy", self.train_env, **self.merged_hparams)
        else:
            if force_new_training_if_model_exists and path_to_load_model and os.path.exists(path_to_load_model):
                self.model_save_path = self.get_next_version_save_path("ppo")
            elif path_to_load_model and not os.path.exists(path_to_load_model):
                self.model_save_path = self.get_next_version_save_path("ppo")
            else:
                self.model_save_path = self.get_model_save_path_for_agent("ppo")
                if os.path.exists(self.model_save_path) and not force_new_training_if_model_exists:
                     self.model_save_path = self.get_next_version_save_path("ppo")
            print(f"  Creating new PPO model. It will be saved to: {os.path.basename(self.model_save_path)}")
            self.model = SB3_PPO("CnnPolicy", self.train_env, **self.merged_hparams)

        # PPO is on-policy. total_timesteps is the main driver.
        # episodes * max_steps_per_episode gives a rough single-environment equivalent.
        # For SB3 VecEnv, total experience is roughly episodes * max_steps_per_episode * n_envs,
        # but SB3 `learn` uses its own `n_steps` per environment to collect rollouts.
        # We'll calculate total_timesteps based on single-env episodes for user understanding.
        # The actual number of environment interactions will be higher due to n_envs.
        # More accurate: training_steps = total_episodes_across_all_envs * average_episode_length
        # Let's use episodes * max_steps as a guideline for total single-agent experience.
        total_timesteps_estimate = episodes * max_steps_per_episode
        # SB3's PPO will run n_steps * n_envs interactions before each update.
        # Total timesteps for model.learn should be significantly higher if using multiple envs.
        # Let's adjust the target for learn() based on n_envs
        adjusted_total_timesteps = total_timesteps_estimate * self.env_creation_params["n_envs"] # This is a very rough guide.
        # A more standard way is to set a very high number of total_timesteps for `learn` and monitor performance.
        # Or calculate based on desired number of updates: num_updates * n_steps * n_envs
        # For simplicity, let's make `episodes` correspond to "epochs" of data collection.
        # If one "episode" for the user means "roughly max_steps_per_episode on one conceptual env", then
        # total_timesteps for model.learn could be episodes * max_steps_per_episode.
        # SB3 will internally distribute this across n_envs.
        print(f"  Target total timesteps for SB3 PPO learn(): ~{total_timesteps_estimate} (distributed across {self.env_creation_params['n_envs']} envs)")


        callbacks = []
        if print_interval_steps > 0: # This interval is for total steps across all envs
            callbacks.append(PrintCallback(print_interval_steps=print_interval_steps))
        if save_interval_eps > 0:
             print(f"  Note: save_interval_eps ({save_interval_eps}) for SB3 PPO is best handled with a custom CheckpointCallback. Model will save at end.")

        try:
            self.model.learn(
                total_timesteps=total_timesteps_estimate, # This means total steps for the entire training process
                callback=callbacks if callbacks else None,
                log_interval=1, # PPO usually logs per rollout completion
                reset_num_timesteps=(path_to_load_model is None)
            )
            print(f"  PPO training loop finished.")
            self.save(self.model_save_path)
        except KeyboardInterrupt:
            print("\n  PPO Training interrupted. Saving current model...")
            self.save(self.model_save_path)
        except Exception as e:
            print(f"  An error occurred during PPO training: {e}")
            import traceback; traceback.print_exc()
            self.save(self.model_save_path) # Attempt to save
        finally:
            if self.train_env: self.train_env.close()
            print(f"--- PPO Training Ended ---")


    def test(self, model_path_to_load, episodes, max_steps_per_episode,
             render_during_test, record_video_flag, video_fps, **kwargs):
        print(f"\n--- PPO Testing ---")
        if not model_path_to_load or not os.path.exists(model_path_to_load):
            print(f"  Error: Model path '{model_path_to_load}' not found. Cannot test PPO.")
            if self.mode == 'test' and model_path_to_load is None:
                print(f"  Attempting to test with a new, untrained PPO model instance.")
                temp_eval_env = self._create_env(for_training=False, n_envs_override=1)
                self.model = SB3_PPO("CnnPolicy", temp_eval_env, **self.merged_hparams)
                temp_eval_env.close()
                model_path_to_load = "Untrained PPO Model"
            else:
                return

        eval_env = gym.make(self.env_id, render_mode="human" if render_during_test else "rgb_array", full_action_space=False)
        eval_env = AtariWrapper(eval_env, clip_reward=False) # No clip for eval
        eval_env = gym.wrappers.FrameStack(eval_env, self.env_creation_params["n_stack"])
        eval_env = Monitor(eval_env)

        if model_path_to_load != "Untrained PPO Model":
            print(f"  Loading PPO model for testing: {os.path.basename(model_path_to_load)}")
            # PPO models (like A2C) can often be loaded without an env if using standard policies
            self.model = SB3_PPO.load(model_path_to_load, env=None, device=self.merged_hparams["device"])
        
        all_rewards = []
        for i in range(episodes):
            obs, info = eval_env.reset()
            terminated, truncated = False, False
            episode_reward, current_steps = 0, 0
            while not (terminated or truncated) and current_steps < max_steps_per_episode:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                current_steps += 1
                if render_during_test and eval_env.render_mode == "human": eval_env.render()
            actual_ep_reward = info.get('episode', {}).get('r', episode_reward)
            all_rewards.append(actual_ep_reward)
            print(f"  PPO Test Episode {i+1}/{episodes} - Score: {actual_ep_reward:.0f}, Steps: {current_steps}")
        eval_env.close()

        avg_reward = np.mean(all_rewards) if all_rewards else 0
        std_reward = np.std(all_rewards) if all_rewards else 0
        print(f"\n  PPO Test Summary: Avg Score: {avg_reward:.2f} +/- {std_reward:.2f}")

        if record_video_flag and model_path_to_load != "Untrained PPO Model":
            print(f"\n  Recording PPO representative run (video)...")
            video_record_env = gym.make(self.env_id, render_mode="rgb_array", full_action_space=False)
            video_record_env = AtariWrapper(video_record_env, clip_reward=False)
            video_record_env = gym.wrappers.FrameStack(video_record_env, self.env_creation_params["n_stack"])
            ts = time.strftime("%Y%m%d_%H%M%S")
            video_folder = os.path.join(self.gifs_dir if self.gifs_dir else "videos", f"ppo_test_{ts}")
            video_env_instance = RecordVideo(
                video_record_env, video_folder=video_folder,
                name_prefix=f"ppo_run_{os.path.basename(model_path_to_load).replace(self.get_model_file_extension(),'')}",
                episode_trigger=lambda ep_id: ep_id == 0, video_length=max_steps_per_episode, fps=video_fps
            )
            try:
                obs, info = video_env_instance.reset()
                terminated, truncated = False, False; vid_ep_reward = 0; vid_steps = 0
                while not (terminated or truncated) and vid_steps < max_steps_per_episode:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = video_env_instance.step(action)
                    vid_ep_reward += reward; vid_steps += 1
                print(f"  PPO Video recorded. Score: {vid_ep_reward:.0f}. Saved in: {video_folder}")
            except Exception as e_vid: print(f"  Error during PPO video recording: {e_vid}")
            finally:
                if video_env_instance: video_env_instance.close()
        print(f"--- PPO Testing Ended ---")


    def evaluate(self, model_path_to_load, episodes, max_steps_per_episode, **kwargs):
        print(f"\n--- PPO Evaluation ---")
        if not model_path_to_load or not os.path.exists(model_path_to_load):
            print(f"  Error: Model path '{model_path_to_load}' not found for PPO. Cannot evaluate.")
            return {}
        
        # Use a VecEnv for evaluate_policy, but can be n_envs=1
        eval_vec_env = self._create_env(for_training=False, n_envs_override=1)
        results = {}
        try:
            self.model = SB3_PPO.load(model_path_to_load, env=eval_vec_env, device=self.merged_hparams["device"]) # Pass env for structure
            print(f"  PPO Model loaded for evaluation: {os.path.basename(model_path_to_load)}")
            
            # evaluate_policy provides a good summary
            # mean_reward_sb3, std_reward_sb3 = evaluate_policy(
            #    self.model, eval_vec_env, n_eval_episodes=episodes, deterministic=True, render=False, warn=False
            # )

            # Manual loop for more detailed stats if needed
            all_ep_rewards, all_ep_steps = [], []
            print(f"  Running manual loop for detailed PPO stats over {episodes} episodes...")
            for i in range(episodes):
                obs = eval_vec_env.reset() # VecEnv obs is a list/array
                terminated, truncated = np.array([False]), np.array([False]) # For VecEnv
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
                if (i + 1) % max(1, episodes // 5) == 0: # Print progress
                    print(f"    PPO Eval Ep {i+1}/{episodes}: Score={actual_ep_reward:.0f}, Steps={actual_ep_steps}")

            avg_s, std_s = (np.mean(all_ep_rewards), np.std(all_ep_rewards)) if all_ep_rewards else (0,0)
            min_s, max_s = (np.min(all_ep_rewards), np.max(all_ep_rewards)) if all_ep_rewards else (0,0)
            avg_st = np.mean(all_ep_steps) if all_ep_steps else 0
            
            # print(f"  PPO Eval (evaluate_policy): Mean Score: {mean_reward_sb3:.2f} +/- {std_reward_sb3:.2f}")
            print(f"  PPO Eval (manual loop): Avg Score: {avg_s:.2f} +/- {std_s:.2f}")
            print(f"    Min: {min_s:.2f}, Max: {max_s:.2f}, Avg Steps: {avg_st:.1f}")

            results = {
                "num_episodes_eval": episodes, "avg_score": round(avg_s, 2),
                "std_dev_score": round(std_s, 2), "min_score": round(min_s, 2),
                "max_score": round(max_s, 2), "avg_steps": round(avg_st, 1)
            }
        except Exception as e:
            print(f"  Error during PPO evaluation: {e}")
            import traceback; traceback.print_exc()
        finally:
            if eval_vec_env: eval_vec_env.close()
        print(f"--- PPO Evaluation Ended ---")
        return results


    def choose_action(self, observation, deterministic=False):
        if self.model is None:
            if self.action_size is None: return 0 # Fallback
            return np.random.choice(self.action_size)
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path):
        if self.model and path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.model.save(path)
                print(f"  PPO Agent (SB3) model saved to {os.path.basename(path)}")
            except Exception as e: print(f"  Error saving PPO model to {path}: {e}")
        elif not self.model: print("  PPO save attempt: model not initialized.")
        elif not path: print("  PPO save attempt: no path provided.")

    def load(self, path):
        if path and os.path.exists(path):
            try:
                print(f"  PPO Agent (SB3) loading model from {os.path.basename(path)}.")
                # PPO models can often be loaded without an env if standard CnnPolicy
                self.model = SB3_PPO.load(path, device=self.merged_hparams["device"])
                print(f"  PPO Model loaded successfully.")
            except Exception as e:
                print(f"  Error loading PPO model from {path}: {e}")
                import traceback; traceback.print_exc()
                self.model = None
        elif not path: print("  PPO load: path is None.")
        else: print(f"  PPO load: path does not exist: {path}")